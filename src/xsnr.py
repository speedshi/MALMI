'''
Estimate the SNR of seismic signals.

Autor: Peidong Shi
Contact: speedshi@hotmail.com  or  peidong.shi@sed.ethz.ch
Create time: 20221110
'''


from obspy import UTCDateTime
import numpy as np


def estimate_snr(trace, stime, noise_window, signal_window, method='maxamp'):
    # trace is one or three component seismic data recroded at a certain station; no gaps in traces;
    # 'stime' is the picked signal arrival-time (phase arrival-time) in datetime format, such as datetime.datetime(2022,2,3,15,21,30,345332).
    # 'noise_window' specify the time window in second relative to 'stime' for evaluating noise, such as [-4, -2].
    # 'signal_window' specify the time window in second relative to 'stime' for evaluating signal, such as [-0.4, 0.6].
    # 'method' specify how to calculate snr; 'maxamp': maximum amplitude (on an arbitrary channel) ratio; 
    #                                        'maxeng': maximum energy ratio (need more than 2 channel);
    #                                         'std': standard deviation ratio;

    if trace.count() == 0:
        # no input seismic data
        return None
    else:
        assert(trace.count()<=3)

        # check traces are coming from the same station
        for itrace in trace:
            assert(itrace.stats.station == trace[0].stats.station)

        stime = UTCDateTime(stime)

        assert(len(noise_window)==2 and (noise_window[0]<=noise_window[1]))
        assert(len(signal_window)==2 and (signal_window[0]<=signal_window[1]))

        noise_start = stime + noise_window[0]
        noise_end = stime + noise_window[1]
        noises = (trace.copy()).trim(starttime=noise_start, endtime=noise_end, pad=True, fill_value=0)  # noise segments

        signal_start = stime + signal_window[0]
        signal_end = stime + signal_window[1]
        signals = (trace.copy()).trim(starttime=signal_start, endtime=signal_end, pad=True, fill_value=0)  # signal segments

        if method.lower() == 'maxeng':
            if noises.count() == 1:
                maxeng_noise = max(np.sqrt(noises[0].data*noises[0].data))
            elif noises.count() == 2:
                maxeng_noise = max(np.sqrt(noises[0].data*noises[0].data + noises[1].data*noises[1].data))
            elif noises.count() == 3:
                maxeng_noise = max(np.sqrt(noises[0].data*noises[0].data + noises[1].data*noises[1].data + noises[2].data*noises[2].data))
            else:
                raise ValueError

            if signals.count() == 1:
                maxeng_signal = max(np.sqrt(signals[0].data*signals[0].data))
            elif signals.count() == 2:
                maxeng_signal = max(np.sqrt(signals[0].data*signals[0].data + signals[1].data*signals[1].data))
            elif signals.count() == 3:
                maxeng_signal = max(np.sqrt(signals[0].data*signals[0].data + signals[1].data*signals[1].data + signals[2].data*signals[2].data))
            else:
                raise ValueError

            snr = maxeng_signal / maxeng_noise
        elif method.lower() == 'maxamp':
            maxamp_noise = np.amax(np.absolute(noises.max()))
            maxamp_signal = np.amax(np.absolute(signals.max()))
            snr = maxamp_signal / maxamp_noise
        elif method.lower() == 'std':
            std_noise = np.amax(noises.std())
            std_signal = np.amax(signals.std())
            snr = std_signal / std_noise
        else:
            raise ValueError("Unrecognized input for method: {method}.")
        
        del noises, signals
    return snr



