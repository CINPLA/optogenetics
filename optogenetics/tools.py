import numpy as np
import quantities as pq
import neo


def make_spiketrain_trials(spike_train, epoch, t_start=None, t_stop=None,
                           dim=None):
    '''
    Makes trials based on an Epoch and given temporal bound

    Parameters
    ----------
    spike_train : neo.SpikeTrain, neo.Unit, numpy.array, quantities.Quantity
    epoch : neo.Epoch
    t_start : quantities.Quantity
        time before epochs, default is 0 s
    t_stop : quantities.Quantity
        time after epochs default is duration of epoch
    dim : str
        if spike_train is numpy.array, the unit must be provided, e.g. "s"

    Returns
    -------
    out : list of neo.SpikeTrains
    '''

    if isinstance(spike_train, neo.Unit):
        sptr = []
        dim = unit.spiketrains[0].times.dimensionality
        unit = unit.spiketrains[0].times.units
        for st in unit.spiketrains:
            sptr.append(spike_train.rescale(dim).magnitude)
        sptr = np.sort(sptr) * unit
    elif isinstance(spike_train, neo.SpikeTrain):
        sptr = spike_train.times
        dim = sptr.dimensionality
        unit = sptr.units
    elif isinstance(spike_train, pq.Quantity):
        assert is_quantities(spike_train, 'vector')
        sptr = spike_train
        dim = sptr.dimensionality
        unit = sptr.units
    elif isinstance(spike_train, np.array):
        sptr = spike_train * pq.Quantity(1, unit)
        dim = sptr.dimensionality
        unit = sptr.units
    else:
        raise TypeError('Expected (neo.Unit, neo.SpikeTrain, ' +
                        'quantities.Quantity, numpy.array), got "' +
                        str(type(spike_train)) + '"')

    from neo.core import SpikeTrain
    if t_start is None:
        t_start = 0 * unit
    if t_start.ndim == 0:
        t_starts = t_start * np.ones(len(epoch.times))
    else:
        t_starts = t_start
        assert len(epoch.times) == len(t_starts), 'epoch.times and t_starts have different size'
    if t_stop is None:
        t_stop = epoch.durations
    if t_stop.ndim == 0:
        t_stops = t_stop * np.ones(len(epoch.times))
    else:
        t_stops = t_stop
        assert len(epoch.times) == len(t_stops), 'epoch.times and t_stops have different size'


    if not isinstance(epoch, neo.Epoch):
        raise TypeError('Expected "neo.Epoch" got "' + str(type(epoch)) + '"')

    trials = []
    for j, t in enumerate(epoch.times.rescale(dim)):
        t_start = t_starts[j].rescale(dim)
        t_stop = t_stops[j].rescale(dim)
        spikes = []
        for spike in sptr[(t+t_start < sptr) & (sptr < t+t_stop)]:
            spikes.append(spike-t)
        trials.append(SpikeTrain(times=spikes * unit,
                                 t_start=t_start,
                                 t_stop=t_stop))
    return trials
