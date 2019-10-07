import numpy as np
import neo
from .tools import make_spiketrain_trials
import quantities as pq
from elephant.spike_train_surrogates import dither_spikes


def get_limit(spike_times, t_start, t_stop):
    if t_start is None:
        try:
            t_start = float(spike_times.t_start)
        except AttributeError:
            t_start = 0

    if t_stop is None:
        try:
            t_stop = float(spike_times.t_stop)
        except AttributeError:
            t_stop = max(spike_times)

    return t_start, t_stop


def histogram(val, bins, density=False):
    '''Fast histogram
    Assuming:
        val, bins are sorted
        bins increase monotonically and uniformly
        all(bins[0] <= v <= bins[-1] for v in val)
    '''
    result = np.zeros(len(bins) - 1).astype(int)
    search = np.searchsorted(bins, val, side='right')
    cnt = np.bincount(search)[1:len(result)+1]
    result[:len(cnt)] = cnt
    if density:
        db = np.array(np.diff(bins), float)
        return result / db / result.sum(), bins
    return result, bins


def stimulus_response_latency_shuffle(spike_times, stim_times, window, binsize, t_start=None, t_stop=None, n_shuffle=1000, plot=False):
    t_start, t_stop = get_limit(spike_times, t_start, t_stop)
    null_trains = dither_spikes(spike_times, window * pq.s, n_shuffle)
    spike_times = np.array(spike_times)
    stim_times = np.array(stim_times)
    n_spikes, n_stim = len(spike_times), len(stim_times)
    bins = np.arange(0, window + binsize, binsize)
    idxs = np.searchsorted(spike_times, stim_times, side='right')
    true_hist, _, = histogram(spike_times[idxs] - stim_times, bins=bins, density=True)
    null_hists = []
    for null_spikes in null_trains:
        null_spikes = np.array(null_spikes)
        idxs = np.searchsorted(null_spikes, stim_times, side='right')
        hist, _ = histogram(null_spikes[idxs] - stim_times, bins=bins, density=True)
        null_hists.append(hist)
    null_hists = np.array(null_hists)
    p_excited, p_inhibited = [], []
    for true, col in zip(true_hist, null_hists.T):
        p_excited.append(sum(col > true) / n_shuffle)
        p_inhibited.append(sum(col < true) / n_shuffle)
    if plot:
        import matplotlib.pyplot as plt
        percentile = 99
        plt.step(bins[1:], true_hist)
        plt.step(bins[1:], np.percentile(null_hists, 100-percentile, 0))
        plt.step(bins[1:], np.percentile(null_hists, percentile, 0))
        plt.step(bins[1:], np.percentile(null_hists, 50, 0))
    return bins[1:] - binsize, true_hist, np.array(p_excited), np.array(p_inhibited)

def stimulus_response_latency_shuffle_kde(spike_times, stim_times, window, dither=30e-3, std=0.01, n_shuffle=100, plot=False):
    from scipy.stats import gaussian_kde
    null_trains = dither_spikes(spike_times, dither * pq.s, n_shuffle)
    spike_times = np.array(spike_times)
    stim_times = np.array(stim_times)
    n_spikes, n_stim = len(spike_times), len(stim_times)
    times = np.arange(-2e-3, window + 1e-4, 1e-4)
    idxs = np.searchsorted(spike_times, stim_times, side='right')
    spikes = np.sort(np.concatenate([spike_times[idxs] - stim_times, spike_times[idxs-1] - stim_times]))
    # spikes = spike_times[idxs] - stim_times
    spikes = spikes[(spikes > times.min()) & (spikes < times.max())]
    true_hist = gaussian_kde(spikes, std)(times)
    null_hists = []
    for null_spikes in null_trains:
        null_spikes = np.array(null_spikes)
        idxs = np.searchsorted(null_spikes, stim_times, side='right')
        spikes = np.sort(np.concatenate([null_spikes[idxs] - stim_times, null_spikes[idxs-1] - stim_times]))
        # spikes = null_spikes[idxs] - stim_times
        spikes = spikes[(spikes > times.min()) & (spikes < times.max())]
        hist = gaussian_kde(spikes, std)(times)
        null_hists.append(hist)
    null_hists = np.array(null_hists)
    p_excited, p_inhibited = [], []
    for true, col in zip(true_hist, null_hists.T):
        p_excited.append(sum(col > true) / n_shuffle)
        p_inhibited.append(sum(col < true) / n_shuffle)
    mask = times >= 0
    times = times[mask]
    true_hist = true_hist[mask]
    p_excited = np.array(p_excited)[mask]
    p_inhibited = np.array(p_inhibited)[mask]

    if plot:
        import matplotlib.pyplot as plt
        percentile = 99
        plt.plot(times, true_hist)
        plt.plot(times, np.percentile(null_hists, 100-percentile, 0)[mask])
        plt.plot(times, np.percentile(null_hists, percentile, 0)[mask])
        plt.plot(times, np.percentile(null_hists, 50, 0)[mask])
    return times, true_hist, p_excited, p_inhibited


def stimulus_response_latency(spike_times, stim_times, window, std, percentile=99, plot=False):
    from scipy.stats import gaussian_kde
    spike_times = np.array(spike_times)
    stim_times = np.array(stim_times)
    n_spikes, n_stim = len(spike_times), len(stim_times)
    times = np.arange(0, window, 1e-4)
    trials = [spike_times[(spike_times >= t - window) & (spike_times <= t + window)] - t
              for t in stim_times]
    spikes = [s for t in trials for s in t]
    kernel = gaussian_kde(spikes, std)

    # we start 10 % away from -window due to edge effects
    pre_times = np.arange(- window + window * 0.1, 0, 1e-4)
    i_percentile = np.percentile(kernel(pre_times), 100 - percentile, 0)
    e_percentile = np.percentile(kernel(pre_times), percentile, 0)
    if plot:
        import matplotlib.pyplot as plt
        all_times = np.arange(-window, window, 1e-4)
        plt.plot(all_times, kernel(all_times))
        plt.plot(pre_times, kernel(pre_times))
        plt.plot(times, [i_percentile] * len(times))
        plt.plot(times, [e_percentile] * len(times))
    return times, kernel, e_percentile, i_percentile


def generate_salt_trials(spike_train, epoch):
    """
    Generate test and baseline trials from spike train and epoch for salt.

    Test trial are trials within epoch times and durations, baseline trails
    are between time + duration and next time.

    Note
    ----
    Spikes before the first trial are disregarded in aseline trials.

    Parameters
    ----------
    spike_train : neo.SpikeTrain
    epoch : neo.Epoch

    Returns
    -------
    out : tuple
        (baseline_trials, test_trials)
    """
    e = epoch
    test_trials = make_spiketrain_trials(spike_train=spike_train,
                                         epoch=e)
    durations = np.array(
        [t2 - t1 - d for t1, t2, d in zip(e.times,
                                          e.times[1:],
                                          e.durations)]) * e.times.units
    times = np.array(
        [t1 + d for t1, d in zip(e.times[:-1], e.durations[:-1])]) * e.times.units
    baseline_epoch = neo.Epoch(times=times, durations=durations)
    baseline_trials = make_spiketrain_trials(spike_train=spike_train,
                                             epoch=baseline_epoch)
    return baseline_trials, test_trials


def salt(baseline_trials, test_trials, winsize, latency_step,
         baseline_t_start, baseline_t_stop, test_t_start, test_t_stop):
    '''SALT   Stimulus-associated spike latency test.
    Calculates a modified version of Jensen-Shannon divergence (see [1]_)
    for spike latency histograms. Please cite [2]_ when using this program.

    Parameters
    ----------
    baseline_trials : Spike raster for stimulus-free baseline
       period. The baseline period has to excede the window size (winsize)
       multiple times, as the length of the baseline segment divided by the
       window size determines the sample size of the null
       distribution (see below).
    test_trials : Spike raster for test period, i.e. after
       stimulus. The test period has to excede the window size (winsize)
       multiple times, as the length of the test period divided by the
       latency_step size determines the number of latencies to be tested.
    winsize : float
        Window size for baseline and test windows.
    latency_step : float
        Step size for test latencies.


    Returns
    -------
    latencies : list
        latencies tested
    p_values : list
        Resulting P values for the Stimulus-Associated spike Latency Test.
    I_values : list
        Test statistic, difference between within baseline and test-to-baseline
        information distance values.

    Notes
    -----
    Briefly, the baseline binned spike raster (baseline_trials) is cut to
    non-overlapping epochs (window size determined by WN) and spike latency
    histograms for first spikes are computed within each epoch. A similar
    histogram is constructed for the test epoch (test_trials). Pairwise
    information distance measures are calculated for the baseline
    histograms to form a null-hypothesis distribution of distances. The
    distances of the test histogram and all baseline histograms are
    calculated and the median of these values is tested against the
    null-hypothesis distribution, resulting in a p value (P).

    References
    ----------
    .. [1] res DM, Schindelin JE (2003) A new metric for probability
       distributions. IEEE Transactions on Information Theory 49:1858-1860.

    .. [2] Kvitsiani D*, Ranade S*, Hangya B, Taniguchi H, Huang JZ, Kepecs A
       (2013) Distinct behavioural and network correlates of two interneuron
       types in prefrontal cortex. Nature 498:363?6.'''
    windows = np.arange(baseline_t_start, baseline_t_stop + winsize, winsize)
    binsize = winsize / 20
    bins = np.arange(- binsize, winsize + binsize, binsize)
    # Latency histogram - baseline
    nbtrials = len(baseline_trials)  # number of trials and number of baseline (pre-stim) data points
    nbins = len(bins)   # number of bins for latency histograms
    nwins = len(windows)
    hlsi = np.zeros((nbins - 1, nwins))   # preallocate latency histograms
    nhlsi = np.zeros((nbins - 1, nwins))    # preallocate latency histograms
    for i in range(nwins - 1):   # loop through baseline windows
        min_spike_times = []
        for j, trial in enumerate(baseline_trials):   # loop through trials
            trial = np.array(trial)
            mask = (trial < windows[i + 1]) & (trial > windows[i])
            spikes_in_win = trial[mask]
            if len(spikes_in_win) > 0:
                min_spike_times.append(spikes_in_win.min() - windows[i])   # latency from window
            else:
                min_spike_times.append(- binsize / 2)   # 0 if no spike in the window
        hlsi[:, i], _ = np.histogram(min_spike_times, bins)   # latency histogram
        nhlsi[:, i] = hlsi[:, i] / sum(hlsi[:, i])   # normalized latency histogram

    latencies = np.arange(test_t_start, test_t_stop + latency_step, latency_step)
    p_values = []
    I_values = []
    nttrials = len(test_trials)   # number of trials
    lsi_tt = np.zeros((nttrials,1)) * np.nan   # preallocate latency matrix
    for latency in latencies:
        min_spike_times = []
        for j, trial in enumerate(test_trials):   # loop through trials
            trial = np.array(trial)
            mask = (trial < latency + winsize) & (trial > latency)
            spikes_in_win = trial[mask]
            if len(spikes_in_win) > 0:
                min_spike_times.append(spikes_in_win.min() - latency)   # latency from window
            else:
                min_spike_times.append(- binsize / 2)   # 0 if no spike in the window
        hlsi[:, nwins - 1], _ = np.histogram(min_spike_times, bins)   # latency histogram
        nhlsi[:, nwins - 1] = hlsi[:, nwins - 1] / sum(hlsi[:, nwins - 1])   # normalized latency histogram
        # JS-divergence
        kn = nwins   # number of all windows (nwins baseline win. + 1 test win.)
        jsd = np.zeros((kn, kn)) * np.nan
        for k1 in range(kn):
            D1 = nhlsi[:, k1]  # 1st latency histogram
            for k2 in range(k1+1, kn):
                D2 = nhlsi[:, k2]   # 2nd latency histogram
                jsd[k1, k2] = np.sqrt(JSdiv(D1, D2) * 2)  # pairwise modified JS-divergence (real metric!)

        # Calculate p-value and information difference
        p, I = makep(jsd, kn)
        p_values.append(p)
        I_values.append(I)
    return latencies, np.array(p_values), np.array(I_values)


def makep(kld, kn):
    '''Calculates p value from distance matrix.'''

    pnhk = kld[:kn - 1, :kn - 1]
    nullhypkld = pnhk[np.isfinite(pnhk)]   # nullhypothesis
    testkld = np.median(kld[:kn - 1, kn - 1])  # value to test
    sno = len(nullhypkld)   # sample size for nullhyp. distribution
    p_value = sum(nullhypkld >= testkld) / sno
    Idiff = testkld - np.median(nullhypkld)   # information difference between baseline and test min_spike_times
    return p_value, Idiff


def JSdiv(P, Q):
    '''JSDIV   Jensen-Shannon divergence.
    Calculates the Jensen-Shannon divergence of the two
    input distributions.'''
    assert abs(sum(P)-1) < 0.00001 or abs(sum(Q)-1) < 0.00001,\
        'Input arguments must be probability distributions.'

    assert P.size == Q.size, 'Input distributions must be of the same size.'

    # JS-divergence
    M = (P + Q) / 2
    D1 = KLdist(P, M)
    D2 = KLdist(Q, M)
    D = (D1 + D2) / 2
    return D


def KLdist(P, Q):
    '''KLDIST   Kullbach-Leibler distance.
    Calculates the Kullbach-Leibler distance (information
    divergence) of the two input distributions.'''
    assert abs(sum(P)-1) < 0.00001 or abs(sum(Q)-1) < 0.00001,\
        'Input arguments must be probability distributions.'

    assert P.size == Q.size, 'Input distributions must be of the same size.'

    # KL-distance
    P2 = P[P * Q > 0]     # restrict to the common support
    Q2 = Q[P * Q > 0]
    P2 = P2 / sum(P2)  # renormalize
    Q2 = Q2 / sum(Q2)

    D = sum(P2 * np.log(P2 / Q2))
    return D
