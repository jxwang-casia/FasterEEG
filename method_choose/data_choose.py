from __future__ import print_function, division

from torch.utils.data import DataLoader

import torch
import numpy as np
import random
import shutil
import inspect
from braindecode.datasets.moabb import MOABBDataset
from braindecode.datautil.preprocess import exponential_moving_standardize
from braindecode.datautil.preprocess import MNEPreproc, NumpyPreproc, preprocess
from braindecode.datautil.windowers import create_windows_from_events


def data_choose(args, block):
    if args.data == 'bci':
        workers = args.worker
        dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[args.data_param.subject_id])#HGD BNCI2014001
        preprocessors = [
            # keep only EEG sensors
            MNEPreproc(fn='pick_types', eeg=True, meg=False, stim=False),
            # convert from volt to microvolt, directly modifying the numpy array
            NumpyPreproc(fn=lambda x: x * 1e6),
            # bandpass filter
            MNEPreproc(fn='filter', l_freq=args.data_param.low_cut_hz, h_freq=args.data_param.high_cut_hz),
            # exponential moving standardization
            NumpyPreproc(fn=exponential_moving_standardize, factor_new=args.data_param.factor_new,
                         init_block_size=args.data_param.init_block_size)
        ]
        # Transform the data
        preprocess(dataset, preprocessors)
        sfreq = dataset.datasets[0].raw.info['sfreq']
        assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

        # Calculate the trial start offset in samples.
        trial_start_offset_samples = int(args.data_param.trial_start_offset_seconds * sfreq)
        # Create windows using braindecode function for this. It needs parameters to define how
        # trials should be used.
        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            window_size_samples=args.model_param.input_window_samples,
            window_stride_samples=args.data_param.n_preds_per_input,
            drop_last_window=False,
            preload=True,
        )
        splitted = windows_dataset.split('session')
        data_set_train = splitted['session_T']
        data_set_val = splitted['session_E']
        data_set_train.sample_name = [str(i) for i in range(len(data_set_train))]
        data_set_val.sample_name = [str(i) for i in range(len(data_set_val))]
        # data_set_train = NTU_SKE(**args.data_param.train_data_param)
        # data_set_val = NTU_SKE(**args.data_param.val_data_param)
        cf = cfv = None
        shutil.copy2(inspect.getfile(MOABBDataset), args.model_saved_name)
    else:
        raise (RuntimeError('No data loader'))

    def init_worker_seed(_):
        torch.cuda.manual_seed_all(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    data_loader_val = DataLoader(data_set_val, batch_size=args.batch_size, shuffle=False,
                                 num_workers=workers, drop_last=False, pin_memory=args.pin_memory,
                                 worker_init_fn=init_worker_seed, collate_fn=cfv)
    data_loader_train = DataLoader(data_set_train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=workers, drop_last=True, pin_memory=args.pin_memory,
                                   worker_init_fn=init_worker_seed, collate_fn=cf)

    block.log('Data load finished: ' + args.data)

    shutil.copy2(__file__, args.model_saved_name)
    return data_loader_train, data_loader_val
