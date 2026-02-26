from __future__ import print_function, division
import shutil
import inspect
from model import *
from braindecode.models import *
from braindecode.models.util import to_dense_prediction_model, get_output_shape


def model_choose(args, block):
    m = args.model
    if m == 'shallow':
        model = ShallowFBCSPNet(n_classes=args.class_num, **args.model_param)
        to_dense_prediction_model(model)
        args.data_param.n_preds_per_input = get_output_shape(model, args.model_param.in_chans, args.model_param.input_window_samples)[2]
        shutil.copy2(inspect.getfile(ShallowFBCSPNet), args.model_saved_name)
    elif m == 'deep':
        model = Deep4Net(n_classes=args.class_num, **args.model_param)
        to_dense_prediction_model(model)
        args.data_param.n_preds_per_input = get_output_shape(model, args.model_param.in_chans, args.model_param.input_window_samples)[2]
        shutil.copy2(inspect.getfile(ShallowFBCSPNet), args.model_saved_name)
    elif m == 'single_shallow':
        model = Single_Shallow(n_classes=args.class_num, **args.model_param)
        if args.model_param['model_type'] == 'tcnfusion':
            args.data_param.n_preds_per_input = 1
        else:
            output_shape = get_output_shape(model, args.model_param.in_chans, args.model_param.input_window_samples)
            if len(output_shape) == 3:
                args.data_param.n_preds_per_input = output_shape[2]
            else:
                args.data_param.n_preds_per_input = 1
        shutil.copy2(inspect.getfile(Single_Shallow), args.model_saved_name)
    elif m == 'adapt_shallow':
        model = Adapt_Shallow(n_classes=args.class_num, args=args, **args.model_param)
        # to_dense_prediction_model(model)
        args.data_param.n_preds_per_input = model.n_preds_per_input
        shutil.copy2(inspect.getfile(Adapt_Shallow), args.model_saved_name)
    else:
        raise (RuntimeError("No modules"))

    shutil.copy2(__file__, args.model_saved_name)
    block.log('Model load finished: ' + args.model)

    return model
