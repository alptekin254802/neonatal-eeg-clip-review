"""Reproduce clip selection, infant-level inference and manuscript Tables 1 and 2.

Inputs are the three original one-second annotation CSVs, per-infant network
scores (NPZ keys time_s and scores), and the published channel annotation XLSX.
All generated files are aggregate results; no infant-level results are written.
"""
import argparse
from hashlib import sha256
from math import comb
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, maximum_filter1d

METHODS = ('isaev', 'uniform_expectation', 'first_mark_proximity')
CATEGORIES = ('target', 'never_marked_by_either_other', 'one_distinct_other_reader',
              'two_readers_disjoint_only', 'simultaneous_other_marks_not_remote_target')


def intervals(mask):
    edges = np.diff(np.r_[0, np.asarray(mask, dtype=int), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def participant_groups():
    ordered = sorted(range(1, 80), key=lambda i: sha256(f'neonatal-second-read-v1:{i}'.encode()).hexdigest())
    return {'development': ordered[:40], 'holdout': sorted(set(range(1, 80)) - set(ordered[:40]))}


def read_scores(directory, infant, length):
    path = directory / f'{infant}.npz'
    with np.load(path) as data:
        times = data['time_s'].astype(float).ravel()
        values = data['scores'].astype(float).ravel()
    if len(times) != len(values):
        raise ValueError(f'Mismatched time and score lengths: {path}')
    if np.isinf(values).any():
        raise ValueError(f'Infinite scores: {path}')
    finite = np.flatnonzero(np.isfinite(values))
    if (not len(finite) or finite[0] != 0 or
            not np.array_equal(finite, np.arange(finite[-1] + 1)) or
            len(values) - 1 - finite[-1] > 8):
        raise ValueError(f'Unexpected missing-score pattern: {path}')
    times, values = times[:finite[-1] + 1], values[:finite[-1] + 1]
    if not (len(values) > 1 and np.array_equal(times, np.arange(len(values))) and
            len(values) >= length - 8):
        raise ValueError(f'Expected a one-second score trace covering the recording: {path}')
    return values[np.minimum(np.arange(length), len(values) - 1)], max(0, length-len(values))



def evaluate_role(infant, role, traces, seconds):
    first = traces[role]
    marked = maximum_filter1d(first.astype(np.uint8), 61, mode='constant').astype(bool)
    starts = np.asarray([s for a, b in intervals(~marked) for s in range(a, b-29, 30)], dtype=int)
    budget = len(starts)//20
    others = np.delete(traces, role, axis=0)
    reference = [(a,b) for a,b in intervals(others.all(axis=0)) if b-a >= 10]
    events = [(a,b) for a,b in reference if not marked[a:b].any()]
    hits = np.asarray([(starts < b) & (starts+30 > a) for a,b in events], dtype=bool).reshape(len(events), len(starts))
    occupied = hits.any(axis=0)
    expected = sum(1-comb(len(starts)-int(row.sum()), budget)/comb(len(starts), budget)
                   for row in hits) if budget else 0.
    common = dict(infant=infant, reader='ABC'[role], eligible=len(starts), selected=budget,
                  reference=len(reference), remote=len(events), reachable=int(hits.any(axis=1).sum()) if budget else 0)
    rows = [dict(**common, method='uniform_expectation', captured=expected,
                 target_free=budget*(1-occupied.mean()) if budget else 0)]
    proximity = -distance_transform_edt(~first) if first.any() else np.zeros(len(first))
    ties = [sha256(f'{infant}:{role}:{s}'.encode()).digest() for s in starts]
    selections = {}
    for name, score in [('first_mark_proximity', [proximity[s:s+30].max() for s in starts]),
                        ('isaev', [seconds[s:s+30].mean() for s in starts])]:
        chosen = sorted(range(len(starts)), key=lambda j: (-score[j], ties[j]))[:budget]
        selections[name] = chosen
        rows.append(dict(**common, method=name, captured=int(hits[:,chosen].any(axis=1).sum()),
                         target_free=int(budget-occupied[chosen].sum())))
    chosen = selections['isaev']
    shown = np.zeros(len(first), dtype=bool)
    for j in chosen:
        shown[starts[j]:starts[j]+30] = True
    event_details = [dict(infant=infant, start=int(a), end=int(b), captured=bool(hits[q,chosen].any()))
                     for q,(a,b) in enumerate(events)]
    duration = {'captured': [], 'missed': []}
    ten_visible = 0
    for event in event_details:
        a,b = event['start'],event['end']
        duration['captured' if event['captured'] else 'missed'].append(b-a)
        ten_visible += int(max((y-x for x,y in intervals(shown[a:b])), default=0) >= 10)
    selected_categories = dict.fromkeys(CATEGORIES,0)
    for j in chosen:
        s = starts[j]
        seg = others[:,s:s+30]
        anyby = seg.any(axis=1)
        if occupied[j]:
            category = 'target'
        elif not anyby.any():
            category = 'never_marked_by_either_other'
        elif anyby.sum() == 1:
            category = 'one_distinct_other_reader'
        elif not seg.all(axis=0).any():
            category = 'two_readers_disjoint_only'
        else:
            category = 'simultaneous_other_marks_not_remote_target'
        selected_categories[category] += 1
    return rows, event_details, duration, ten_visible, selected_categories


def summarize(frame):
    columns = ['captured','remote','selected','target_free','reachable','reference','eligible']
    subjects = frame.groupby(['method','infant'],sort=True)[columns].sum()
    base = subjects.loc['uniform_expectation']
    positive = base.index[base.remote > 0]
    if len(positive) < 2:
        raise ValueError('At least two infants with targets are required for bootstrap inference.')
    boot = np.random.default_rng(181126).integers(len(positive),size=(10000,len(positive)))
    rates = {m:(subjects.loc[m].loc[positive].captured/subjects.loc[m].loc[positive].remote).to_numpy() for m in METHODS}
    result = {}
    for method, rate in rates.items():
        group = subjects.loc[method]
        result[method] = dict(captured=float(group.captured.sum()),remote=int(group.remote.sum()),
            selected=int(group.selected.sum()),target_free=float(group.target_free.sum()),
            infants_with_capture=int((group.captured>0).sum()) if method!='uniform_expectation' else None,
            macro_capture=float(rate.mean()),
            paired_macro_differences={})
        for comparator in METHODS[1:]:
            if method != 'isaev' or method == comparator:
                continue
            d = rate-rates[comparator]
            result[method]['paired_macro_differences'][comparator] = dict(mean=float(d.mean()),percentile95=np.quantile(d[boot].mean(axis=1),[.025,.975]).tolist())
    differences = []
    for infant in base.index:
        keep = positive[positive != infant]
        network_rate = subjects.loc['isaev'].loc[keep].captured/base.loc[keep].remote
        uniform_rate = subjects.loc['uniform_expectation'].loc[keep].captured/base.loc[keep].remote
        differences.append(float((network_rate-uniform_rate).mean()))
    influence = {'uniform_expectation': [min(differences), max(differences)]}
    contributor_omissions = {}
    cnn = subjects.loc['isaev']
    for column in ('captured', 'remote'):
        maximum = cnn[column].max()
        records = []
        for infant in cnn.index[cnn[column] == maximum]:
            keep = positive[positive != infant]
            cnn_rate = cnn.loc[keep].captured/base.loc[keep].remote
            uniform_rate = subjects.loc['uniform_expectation'].loc[keep].captured/base.loc[keep].remote
            records.append(dict(removed_targets=int(cnn.loc[infant].remote),
                removed_captures=int(cnn.loc[infant].captured),
                difference_vs_uniform=float((cnn_rate-uniform_rate).mean())))
        contributor_omissions['largest_'+column] = records
    return subjects, dict(infants=len(base),opportunity_infants=len(positive),reachable=int(base.reachable.sum()),
        results=result,leave_one_infant_out_difference_range=influence,largest_contributor_omissions=contributor_omissions)


def quantiles(values):
    return dict(n=len(values), quartiles_seconds=np.quantile(values,[.25,.5,.75]).tolist() if values else None)


def pooled_summary(groups):
    rng = np.random.default_rng(181126)
    draws = np.zeros((10000,len(METHODS)))
    arrays = []
    for subjects in groups:
        base = subjects.loc['uniform_expectation']
        ids = base.index[base.remote > 0]
        a = np.column_stack([(subjects.loc[m].loc[ids].captured/base.loc[ids].remote).to_numpy() for m in METHODS])
        arrays.append(a)
        boot = rng.integers(len(a),size=(10000,len(a)))
        draws += a[boot].sum(axis=1)
    draws /= sum(map(len,arrays))
    all_rates = np.concatenate(arrays)
    result = {method: dict(mean=float(all_rates[:,j].mean()))
              for j,method in enumerate(METHODS[:2])}
    result['isaev_minus_uniform_expectation'] = dict(
        mean=float((all_rates[:,0]-all_rates[:,1]).mean()),
        percentile95=np.quantile(draws[:,0]-draws[:,1],[.025,.975]).tolist())
    result['opportunity_infants'] = int(len(all_rates))
    result['captured'] = int(sum(s.loc['isaev'].captured.sum() for s in groups))
    result['remote'] = int(sum(s.loc['isaev'].remote.sum() for s in groups))
    return result


def write_tables(results, output):
    table1,table2 = {},{}
    for name,r in results.items():
        cnn,u,p = (r['results'][m] for m in METHODS)
        diff = cnn['paired_macro_differences']
        fmt = lambda d: f"{100*d['mean']:.1f} ({100*d['percentile95'][0]:.1f} to {100*d['percentile95'][1]:.1f})"
        table1[name] = {'Infants / infants with targets':f"{r['infants']} / {r['opportunity_infants']}",
            'Target event–reader instances':int(cnn['remote']),'Network captured instances':int(cnn['captured']),
            'Uniform expected captured instances':f"{u['captured']:.2f}",'Proximity captured instances':int(p['captured']),
            'Network / uniform mean capture, %':f"{100*cnn['macro_capture']:.1f} / {100*u['macro_capture']:.1f}",
            'Network minus uniform (95% interval)':fmt(diff['uniform_expectation']),
            'Network minus proximity (95% interval)':fmt(diff['first_mark_proximity']),
            'Captured instances with 10 consecutive seconds displayed':r['ten_visible']}
        count = cnn['selected']; target = count-cnn['target_free']; expected=count-u['target_free']
        cats = r['selected_categories']
        table2[name] = {'Simulated readings':r['infants']*3,'Selected clips':count,
            'Mean displayed EEG per reading, minutes':f"{count*.5/(3*r['infants']):.2f}",
            'Network target-overlapping clips':int(target),'Network target-overlapping clips, %':f'{100*target/count:.2f}',
            'Uniform expected target-overlapping clips':f'{expected:.2f}',
            'Uniform expected target-overlapping clips, %':f'{100*expected/count:.2f}','Clip enrichment, fold':f'{target/expected:.2f}',
            'Target-free clips unmarked by either other expert':cats[CATEGORIES[1]],
            'Target-free clips marked by exactly one other expert':cats[CATEGORIES[2]],
            'Target-free clips with simultaneous marks failing target definition':cats[CATEGORIES[4]]}
    for number,table in [(1,table1),(2,table2)]:
        frame = pd.DataFrame(table).rename_axis('Measure')
        frame.to_csv(output/f'table_{number}.csv')
        lines = ['| Measure | Development | Participant holdout |','|---|---:|---:|']
        lines.extend(f'| {measure} | {row.development} | {row.holdout} |' for measure,row in frame.iterrows())
        (output/f'table_{number}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--scores',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    tables = [pd.read_csv(args.data/f'annotations_2017_{suffix}.csv') for suffix in ('A_fixed','B','C')]
    workbook = args.data/'seizure_per_channel_Dec_2017.xlsx'
    if not workbook.exists():
        raise FileNotFoundError('The published channel-annotation workbook is required for the reported coverage analysis: '+str(workbook))
    channels = pd.read_excel(workbook).dropna(subset=['FILE ID'])
    results,subject_tables = {},[]
    for name,ids in participant_groups().items():
        rows,events = [],[]
        durations = {'captured':[],'missed':[]}
        selected = dict.fromkeys(CATEGORIES,0)
        total_seconds=tail_seconds=ten_visible=0
        for infant in ids:
            series = [table[str(infant)].dropna() for table in tables]
            if not (len({len(s) for s in series})==1 and all(s.index.equals(pd.RangeIndex(len(s))) and s.isin([0,1]).all() for s in series)):
                raise ValueError(f'Invalid annotation coverage or values for infant {infant}')
            traces = np.stack([s.to_numpy(dtype=bool) for s in series])
            seconds,tail = read_scores(args.scores,infant,traces.shape[1])
            total_seconds += traces.shape[1]
            tail_seconds += tail
            for role in range(3):
                new_rows,new_events,duration,ten,cats = evaluate_role(infant,role,traces,seconds)
                rows.extend(new_rows);events.extend(new_events);ten_visible += ten
                for k in durations:
                    durations[k].extend(duration[k])
                for k in CATEGORIES:
                    selected[k]+=cats[k]
        subjects,r = summarize(pd.DataFrame(rows))
        subject_tables.append(subjects)
        overlap = 0
        for event in events:
            subset = channels[channels['FILE ID']==event['infant']]
            found = ((subset.Start < event['end']) & (subset.Stop > event['start'])).any()
            overlap += int(found)
        r.update(hours=total_seconds/3600,carried_forward_tail_seconds=tail_seconds,ten_visible=ten_visible,
            selected_categories=selected,
            event_duration_seconds={k:quantiles(v) for k,v in durations.items()},
            channel_coverage=dict(remote_events=len(events), events_with_channel_annotations=overlap))
        results[name]=r
    args.output.mkdir(parents=True,exist_ok=True)
    write_tables(results,args.output)
    summary = dict(groups=results,pooled_secondary=pooled_summary(subject_tables),
        channel_annotation_source=dict(sha256=sha256(workbook.read_bytes()).hexdigest(),rows=len(channels),infants=int(channels['FILE ID'].nunique())))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('Wrote aggregate summary.json and Tables 1 and 2 to',args.output)


if __name__ == '__main__':
    main()
