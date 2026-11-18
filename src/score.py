"""Run the original Duke-pretrained DL2 on Helsinki EEG, without fitting.

Requires Python 3.11, TensorFlow/Keras 2.15, MATLAB with Signal Processing
Toolbox, and dependencies downloaded by acquire.py. Native epoch outputs retain
source flatline flags; every epoch is scored. Dense scores average overlapping
8-second windows. Uncovered trailing seconds remain NaN in the stored output.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import h5py
import numpy as np
from scipy.signal import lfilter

from acquire import digest, MODEL_SHA256, MATLAB_FILES, WEIGHTS_SHA256


def matlab_quote(path):
    return str(Path(path).resolve()).replace('\\', '/').replace("'", "''")


def extract_epochs(data, infants, destination, matlab='matlab'):
    """Extract every missing recording in one MATLAB process, saving atomically."""
    source = data / 'external' / 'preprocessing'
    destination.mkdir(parents=True, exist_ok=True)
    for name, checksum in MATLAB_FILES.items():
        if digest(source / name) != checksum:
            raise ValueError(f'Preprocessing dependency integrity failure: {name}')
    for infant in infants:
        if not (data / 'edf' / f'eeg{infant}.edf').exists():
            raise FileNotFoundError(data / 'edf' / f'eeg{infant}.edf')
    if not infants:
        return
    code = f"""
addpath('{matlab_quote(source)}');
maxNumCompThreads(1);
notch=load(fullfile('{matlab_quote(source)}','notch_filter.mat'));
for id=[{' '.join(map(str,infants))}]
 target=fullfile('{matlab_quote(destination)}',sprintf('%d.mat',id));
 partial=[target '.partial.mat'];
 [sig,sc,fsorig]=read_data_montage(fullfile('{matlab_quote(data / 'edf')}',sprintf('eeg%d.edf',id)));
 assert(all(fsorig==256)); duration_s=numel(sig{{1}})/256;
for c=1:18
 dat=filter(notch.Num,notch.Den,double(sig{{c}}*sc));
 dat=resample(dat,32,256);
 inds=1:128:length(dat); inds=inds(inds<(length(dat)-255));
 if c==1, epochs=zeros(numel(inds),18,256,'single'); valid=true(numel(inds),1); end
 for j=1:numel(inds)
  epoch=dat(inds(j):inds(j)+255);
  if mean(abs(epoch(1:128)))<1e-4 || mean(abs(epoch(128:end)))<1e-4, valid(j)=false; end
  epochs(j,c,:)=single(epoch)*single(.001);
 end
end
starts_s=(inds-1)/32;
save(partial,'epochs','starts_s','valid','duration_s','-v7.3');
movefile(partial,target,'f');
fprintf('Extracted infant %d: %d epochs\\n',id,numel(inds));
clear sig epochs;
end
"""
    with tempfile.TemporaryDirectory(prefix='eeg-extraction-') as temp:
        script = Path(temp) / 'prepare_eeg_epochs.m'
        script.write_text(code, encoding='utf-8')
        subprocess.run([matlab, '-batch', f"run('{matlab_quote(script)}')"], check=True)


def load_detector(data):
    """Adapt imports only, then load 12-channel weights and copy to 18 channels."""
    import tensorflow as tf
    if not tf.__version__.startswith('2.15.'):
        raise RuntimeError('Validated runtime requires TensorFlow 2.15.x / Keras 2.15')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    external = data / 'external'
    original = external / 'models.py'
    weights = external / 'DL2_Duke_pretrained.model_weights'
    if digest(original) != MODEL_SHA256 or digest(weights) != WEIGHTS_SHA256:
        raise ValueError('Original model/checkpoint integrity failure; rerun acquisition')
    # Adapt import namespaces in memory while preserving layer definitions.
    text = original.read_text(encoding='utf-8').replace('from keras.', 'from tensorflow.keras.').replace('import keras.backend', 'import tensorflow.keras.backend')
    namespace = {'__name__':'isaev_original'}
    exec(compile(text, str(original), 'exec'), namespace)
    origin = namespace['model_DL2'](np.empty((1,12,256)), np.empty((1,1)))
    origin.load_weights(str(weights))
    target = namespace['model_DL2'](np.empty((1,18,256)), np.empty((1,1)))
    namespace['copy_model'](origin, target)
    return target


def dense_scores(starts, predictions, valid, duration):
    """Average all native windows covering each whole elapsed second."""
    total = int(np.floor(duration))
    sums = np.zeros(total)
    counts = np.zeros(total, dtype=np.int32)
    invalid = np.zeros(total, dtype=np.int32)
    for start, probability, usable in zip(starts, predictions, valid):
        a, b = int(start), min(int(start)+8, total)
        sums[a:b] += float(probability)
        counts[a:b] += 1
        invalid[a:b] += int(not usable)
    score = np.divide(sums, counts, out=np.full(total,np.nan), where=counts>0)
    return dict(scores=score, time_s=np.arange(total,dtype=float), epoch_scores=predictions,
                epoch_start_s=starts, epoch_valid=valid, coverage_count=counts,
                invalid_epoch_count=invalid, valid_time=(counts>invalid)&(counts>0),
                duration_s=duration)


def score_file(model, epochs, output):
    with h5py.File(epochs) as source:
        x = source['epochs'][:].transpose(2,1,0)
        starts = source['starts_s'][:].ravel()
        valid = source['valid'][:].ravel().astype(bool)
        duration = float(source['duration_s'][0,0])
    if x.shape != (len(starts),18,256) or len(valid) != len(starts):
        raise ValueError('Unexpected epoch layout')
    x = lfilter([1,-1], [1,-(1-.5*2*np.pi/32)], x, axis=-1).astype('float32')
    predictions = model.predict(x, batch_size=64, verbose=0).ravel()
    if not (np.isfinite(predictions).all() and ((predictions>=0)&(predictions<=1)).all()):
        raise ValueError('Invalid native probabilities')
    arrays = dense_scores(starts, predictions, valid, duration)
    temporary = output.with_name(output.name + '.part')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(output)
    return {'infant':int(output.stem), 'seconds':len(arrays['scores']),
            'epochs':len(predictions), 'source_flagged_epochs':int((~valid).sum()),
            'uncovered_seconds':int((arrays['coverage_count']==0).sum()),
            'sha256':digest(output)}


def check_preprocessing(data, epochs, report):
    """Compare all EEG1 channel inputs and epoch validity with the author example."""
    from scipy.io import loadmat
    from acquire import acquire_reference
    reference_path = acquire_reference(data)
    reference = loadmat(reference_path)['all_feats'][0,0]
    author = np.stack([
        np.stack([v.ravel() for v in reference[c,0]['epochs'][0,0].ravel()])
        for c in range(18)], axis=1) * .001
    with h5py.File(epochs / '1.mat') as source:
        extracted = source['epochs'][:].transpose(2,1,0)
        valid = source['valid'][:].ravel().astype(bool)
    processed = lfilter([1,-1], [1,-(1-.5*2*np.pi/32)], extracted, axis=-1)
    if author.shape != processed.shape:
        raise ValueError('Author example tensor shape differs')
    author_valid = np.isfinite(author).all(axis=(1,2))
    finite = np.isfinite(author)
    difference = (processed-author)[finite]
    result = dict(epochs=len(valid), channels=18, samples_per_epoch=256,
                  author_valid_epochs=int(author_valid.sum()),
                  validity_mismatches=int((valid != author_valid).sum()),
                  finite_rmse=float(np.sqrt(np.mean(difference ** 2))),
                  maximum_absolute_error=float(np.max(np.abs(difference))),
                  reference_sha256=digest(reference_path))
    result['passed'] = result['validity_mismatches'] == 0 and result['maximum_absolute_error'] < 1e-6
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result,indent=2), encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['passed']:
        raise ValueError('Author preprocessing-reference comparison failed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--output', type=Path, default=Path('scores'))
    parser.add_argument('--ids', nargs='+', type=int, default=list(range(1,80)))
    parser.add_argument('--matlab', default='matlab', help='MATLAB executable path')
    parser.add_argument('--epochs', type=Path, help='Reuse MATLAB epoch files named N.mat')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--check-preprocessing', action='store_true',
                        help='Compare EEG1 inputs and source validity with author example, then exit')
    args = parser.parse_args()
    if not set(args.ids) <= set(range(1,80)):
        parser.error('IDs must be 1..79')
    args.data = args.data.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs or args.data / 'epochs'
    epochs.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    if args.check_preprocessing:
        if not (epochs / '1.mat').exists():
            extract_epochs(args.data, [1], epochs, args.matlab)
        check_preprocessing(args.data, epochs, args.output / 'preprocessing_check.json')
        return
    missing = [i for i in args.ids if not (epochs / f'{i}.mat').exists()
               and (args.overwrite or not (args.output / f'{i}.npz').exists())]
    extract_epochs(args.data, missing, epochs, args.matlab)
    model = load_detector(args.data)
    rows = []
    for infant in args.ids:
        output = args.output / f'{infant}.npz'
        receipt = output.with_suffix('.json')
        if output.exists() and receipt.exists() and not args.overwrite:
            prior = json.loads(receipt.read_text(encoding='utf-8'))
            if (prior.get('checkpoint_sha256') == WEIGHTS_SHA256 and
                    prior.get('source_model_sha256') == MODEL_SHA256 and
                    digest(output) == prior.get('sha256')):
                rows.append(prior)
                print(f'Reusing verified scores for infant {infant}', flush=True)
                continue
        epoch_file = epochs / f'{infant}.mat'
        if not epoch_file.exists():
            extract_epochs(args.data, [infant], epochs, args.matlab)
        row = score_file(model, epoch_file, output)
        row.update(checkpoint_sha256=WEIGHTS_SHA256, source_model_sha256=MODEL_SHA256)
        receipt.write_text(json.dumps(row,indent=2), encoding='utf-8')
        rows.append(row)
        print(json.dumps(row), flush=True)
    (args.output / 'run.json').write_text(json.dumps(rows,indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
