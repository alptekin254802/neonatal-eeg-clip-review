"""Download public recordings, annotations and original detector dependencies.

Run: python src/acquire.py --data data
Downloads are checked before reuse; interrupted EDF downloads resume by HTTP Range.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time

import requests

ISAEV_COMMIT = '36f58dc8315a7f9119ebc7220dee419a8b10f3ce'
TAPANI_COMMIT = '1a59e4b414f28aba52e61c3360fdc324199071b4'
MATLAB_FILES = {
    'read_edf.m':'de945314df00cea752e9fb28f55f5a46776c494de2567506db8eff8bdea34d94',
    'read_data_montage.m':'57196447f307d35e981b57fab8179dd9bddde9be02badaad5db654e4adfef4dd',
    'notch_filter.mat':'5a8ea53ec98081127aa78d0648782ed35b29241dfa9baa6c9cbd3dbf206f3e7f',
}
WEIGHTS_SHA256 = 'fda6161481983e2cb5b19be798edbc137fdde5c063a2a647ce172fc3a2e693dc'
MODEL_SHA256 = 'b37244c8cd4a0285bb95bd319af184c4657b0a1bdef2f5062a77ced82c6c6164'
WORKBOOK_SHA256 = '79b3fbf8064da56704a130134dffaaa01f9b01c839814561b7efb89e343bcab2'
ANNOTATIONS = {'A_fixed':'c70e06649d7ddff33913c14dd1691670',
               'B':'91c586a7abe3a02b04731dccd2043866',
               'C':'02e37e87a4c0c96cc8f812a4a0c88ea7'}


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(url, path, expected=None, algorithm='sha256', size=None):
    """Atomically install a checked file; retain partial bytes after network failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (size is None or path.stat().st_size == size):
        if expected is not None and digest(path, algorithm) == expected:
            return path
    partial = path.with_name(path.name + '.part')
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if size is not None and offset >= size:
                partial.unlink()
                offset = 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with requests.get(url, headers=headers, stream=True, timeout=(30,120)) as response:
                response.raise_for_status()
                resume = offset > 0 and response.status_code == 206
                if resume and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                    raise ValueError('Server returned incompatible partial response')
                with partial.open('ab' if resume else 'wb') as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
            if size is not None and partial.stat().st_size != size:
                raise ValueError(f'Incorrect size: {path.name}')
            if expected is not None and digest(partial, algorithm) != expected:
                partial.unlink()
                raise ValueError(f'Checksum mismatch: {path.name}')
            partial.replace(path)
            return path
        except (requests.RequestException, ValueError):
            if attempt == 4:
                raise
            time.sleep(min(16, 2 ** attempt))


def acquire(data, ids=range(1,80), workers=2, metadata_only=False):
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    metadata_path = data / 'zenodo_4940267.json'
    if not metadata_path.exists():
        response = requests.get('https://zenodo.org/api/records/4940267', timeout=90)
        response.raise_for_status()
        metadata_path.write_text(json.dumps(response.json(), indent=2), encoding='utf-8')
    record = json.loads(metadata_path.read_text(encoding='utf-8'))
    if int(record['id']) != 4940267:
        raise ValueError('Expected Helsinki version 4 record 4940267')
    files = {item['key']:item for item in record['files']}
    names = [f'annotations_2017_{role}.csv' for role in ANNOTATIONS]
    if not metadata_only:
        names += [f'eeg{ident}.edf' for ident in ids]
    def retrieve(name):
        entry = files[name]
        algorithm, checksum = entry['checksum'].split(':')
        if name.startswith('annotations_'):
            role = name.removeprefix('annotations_2017_').removesuffix('.csv')
            if checksum != ANNOTATIONS[role]:
                raise ValueError('Published annotation checksum differs from pinned version')
        destination = data / 'edf' / name if name.endswith('.edf') else data / name
        try:
            download(entry['links']['self'], destination, checksum, algorithm, entry['size'])
        except (requests.RequestException, ValueError):
            download(f'https://zenodo.org/records/4940267/files/{name}?download=1',
                     destination, checksum, algorithm, entry['size'])
        print(f'Verified {name}', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(retrieve, names))
    base = f'https://raw.githubusercontent.com/ktapani/Neonatal_Seizure_Detection/{TAPANI_COMMIT}'
    download(base + '/Neonatal_Seizure_Detection_Algorithm/reproducibility/seizure_per_channel_Dec_2017.xlsx',
             data / 'seizure_per_channel_Dec_2017.xlsx', WORKBOOK_SHA256)
    external = data / 'external'
    external.mkdir(exist_ok=True)
    download(f'https://raw.githubusercontent.com/dyisaev/seizure-detection-neonates/{ISAEV_COMMIT}/models.py',
             external / 'models.py', MODEL_SHA256)
    for name, checksum in MATLAB_FILES.items():
        download(base + '/Neonatal_Seizure_Detection_Algorithm/neonatal_sez_det/' + name,
                 external / 'preprocessing' / name, checksum)
    weights = external / 'DL2_Duke_pretrained.model_weights'
    if not weights.exists() or digest(weights) != WEIGHTS_SHA256:
        import gdown
        temporary = weights.with_name(weights.name + '.part')
        gdown.download(id='1uAt9kxTCQLwgU5y7KJXC7AKDYJ8FlOE2', output=str(temporary), quiet=False)
        if digest(temporary) != WEIGHTS_SHA256:
            raise ValueError('Checkpoint SHA256 mismatch')
        temporary.replace(weights)
    (external / 'sources.json').write_text(json.dumps({
        'isaev_commit':ISAEV_COMMIT, 'tapani_commit':TAPANI_COMMIT,
        'checkpoint_sha256':WEIGHTS_SHA256, 'matlab_files_sha256':MATLAB_FILES}, indent=2), encoding='utf-8')


def acquire_reference(data):
    """Retrieve the authors' public EEG1 preprocessing example."""
    target = Path(data) / 'reference' / 'helsinki_eeg1_feats.mat'
    expected = 'e1a3ff41dc465603372e6e4a678257cc30ccf895e78d67b5e79d91162ad06787'
    if target.exists() and digest(target) == expected:
        return target
    import gdown
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    gdown.download(id='1yxDxGO-YETk1wzH-W8RPO4i-6dYwVb46', output=str(partial), quiet=False)
    if digest(partial) != expected:
        raise ValueError('Author example SHA256 mismatch')
    partial.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--ids', type=int, nargs='+', default=list(range(1,80)))
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--metadata-only', action='store_true', help='Skip EDF files')
    parser.add_argument('--reference-example', action='store_true', help='Also download author EEG1 preprocessing example')
    args = parser.parse_args()
    if not set(args.ids) <= set(range(1,80)) or not 1 <= args.workers <= 4:
        parser.error('IDs must be 1..79 and workers 1..4')
    acquire(args.data, args.ids, args.workers, args.metadata_only)
    if args.reference_example:
        acquire_reference(args.data)


if __name__ == '__main__':
    main()
