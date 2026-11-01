
import numpy as np
import warnings
import torch

from model_classes import SimpleForecaster, AutoARIMA, \
    GARCHModel, NBeatsModel, \
        ProphetModel, GRUForecaster, \
            TCNForecaster, LocalDeepARPredictor, GlobalDeepARPredictor

from helpers import evaluate_forecast

def run_bench(
        t2f_predictions,
        train_df,
        test_df,
        target_col,
        covariate_cols,
        weight_col,
        timestamp_col,
        series_id_col,
        hidden_size, #n-beats, gru, deepar
        num_layers, #gru, deepar
        local_length,
        global_length,
        epochs, #n-beats, gru, tcn, deepar
        batch_size, #n-beats, gru, tcn
        logger,
        bench_models:list[str]=['AUTOARIMA',
                        'GARCH',
                        'N-BEATS',
                        'Prophet',
                        'GRU',
                        'TCN',
                        'DeepAR local',
                        'DeepAR global']
        ):

    input_size=len(covariate_cols),  # number of features (feature1, feature2)
    output_size=1  # number of target variables (value), fixed

    X_test = test_df[covariate_cols]
    y_test = test_df[target_col]
    weights_test = test_df[weight_col]

    X_train = train_df[covariate_cols]
    y_train = train_df[target_col]
    covariates_train = train_df[covariate_cols + [weight_col]]

    series_ids = train_df.index.get_level_values(series_id_col).unique()
    
    print(f"Number of series: {len(series_ids)}")
    print(f"Number of covariates: {len(covariates_train)}")
    print(f"Local output length: {local_length}")
    print(f"Global output length: {global_length}")

    # Prepare results storage
    local_results = {
        'T2f': {'mae': [], 'rmse': [], 'smape': [], 'wsmae': []}
    }
    
    global_results = {
        'T2f':  {'true': [], 'pred': [], 'weights': []}
    }

    for m in bench_models:
        local_results[m] = {'mae': [], 'rmse': [], 'smape': [], 'wsmae': []}
        global_results[m] =  {'true': [], 'pred': [], 'weights': []}

    
    # init models
    simple_forecaster = SimpleForecaster(target_col=target_col)

    model_lib = {
        'simple': {'model': simple_forecaster, 'pred': ''}
    }

    def x_pred(model, model_str, train_df, test_df, series_id, target_col, local_length, global_length):
        series_train = train_df.loc[series_id, target_col]
        try:
            model.fit(series_train)
            series_pred = model.predict(steps=global_length)
        except Exception as e:
            warnings.warn(f"{model_str} prediction failed for series {series_id}: {str(e)}. Using last known value.")
            series_pred = np.full(global_length, series_train.iloc[-1])
    
        # Ensure predictions match the expected length
        if len(series_pred) < global_length:
            series_pred = np.pad(series_pred, (0, global_length - len(series_pred)), 'edge')
        elif len(series_pred) > global_length:
            series_pred = series_pred[:global_length]

        return np.array(series_pred)

#     def xy_pred(model, model_str, train_df, test_df, series_id, target_col, local_length, global_length):
#         series_train = train_df.loc[series_id]
#         X_train = series_train[covariate_cols].values
#         y_train = series_train[target_col].values

#         try:
#             model.fit(X_train, y_train)
#             X_test = test_df.loc[series_id, covariate_cols].values[-local_length:]
#             series_pred = model.predict(X_test, steps=global_length)
            
#         except Exception as e:
#             print(f"Error in {model_str} prediction for series {series_id}: {str(e)}. Using last known value.")
#             series_pred = np.full(global_length, series_train.iloc[-1])

#         # Ensure predictions match the expected length
#         if len(series_pred) < global_length:
#             series_pred = np.pad(series_pred, (0, global_length - len(series_pred)), 'edge')
#         elif len(series_pred) > global_length:
#             series_pred = series_pred[:global_length]

#         return np.array(series_pred)

    def xy_pred(model, model_str, train_df, test_df, series_id, target_col, local_length, global_length):
        series_train = train_df.loc[series_id]
        X_train = series_train[covariate_cols].values
        y_train = series_train[target_col].values

        try:
            model.fit(X_train, y_train)
            X_test = test_df.loc[series_id, covariate_cols].values[-local_length:]
            series_pred = model.predict(X_test, steps=global_length)

        except Exception as e:
            print(f"Error in {model_str} prediction for series {series_id}: {str(e)}. Using last known value.")
            # Get last value safely
            last_value = y_train[-1] if len(y_train) > 0 else 0.0
            series_pred = np.full(global_length, last_value, dtype=np.float64)

        # Ensure predictions match the expected length and are the right type
        series_pred = np.array(series_pred, dtype=np.float64)  # Ensure float type
        if len(series_pred) < global_length:
            series_pred = np.pad(series_pred, (0, global_length - len(series_pred)), 'edge')
        elif len(series_pred) > global_length:
            series_pred = series_pred[:global_length]

        return series_pred


    if 'AUTOARIMA' in bench_models:
        autoarima = AutoARIMA()
        model_lib['AUTOARIMA'] = {'model': autoarima, 'pred': x_pred}

    if 'GARCH' in bench_models:
        garch = GARCHModel()
        model_lib['GARCH'] = {'model': garch, 'pred': x_pred}

    if 'N-BEATS' in bench_models:
        nbeats = NBeatsModel(
            input_size=local_length,
            output_size=global_length,
            hidden_size=hidden_size,
            epochs=epochs,
            batch_size=batch_size
        )

        def nbeats_pred(model, model_str, train_df, test_df, series_id, target_col, local_length, global_length):       
            # Prepare data for N-BEATS
            series_train = train_df.loc[series_id, target_col].values
            try:
                # Create sequences for training
                X_train_nbeats = []
                y_train_nbeats = []
                
                for i in range(len(series_train) - local_length - global_length + 1):
                    X_train_nbeats.append(series_train[i:i+local_length])
                    y_train_nbeats.append(series_train[i+local_length:i+local_length+global_length])
                    
                # Skip if not enough data points
                if len(X_train_nbeats) < 2:  # Add this check
                    raise ValueError("Not enough data points for training")
                
                X_train_nbeats = torch.FloatTensor(np.array(X_train_nbeats)).to(nbeats.device)
                y_train_nbeats = torch.FloatTensor(np.array(y_train_nbeats)).to(nbeats.device)
                                
                # Fit N-BEATS model
                nbeats.fit(X_train_nbeats, y_train_nbeats)
                
                # Prepare test data
                series_test = test_df.loc[series_id, target_col].values
                X_test_nbeats = torch.FloatTensor(series_test[:global_length]).reshape(1, -1).to(nbeats.device)
        
                # Use the model's evaluate method
                series_pred = nbeats.evaluate(X_test_nbeats, series_true)
                    
            except Exception as e:
                print(f"Error in {model_str} prediction for series {series_id}: {str(e)}. Using last known value.")
                series_pred = np.full(global_length, series_train[-1])

            # Ensure predictions match the expected length
            if len(series_pred) < global_length:
                series_pred = np.pad(series_pred, (0, global_length - len(series_pred)), 'edge')
            elif len(series_pred) > global_length:
                series_pred = series_pred[:global_length]

            return np.array(series_pred)
            
        model_lib['N-BEATS'] = {'model': nbeats, 'pred': nbeats_pred}

    if 'Prophet' in bench_models:
        prophet = ProphetModel(timestamp_col=timestamp_col, target_col=target_col)

        def prophet_pred(model, model_str, train_df, test_df, series_id, target_col, local_length, global_length):
            # Prophet evaluation
            series_train = train_df.loc[series_id].reset_index()
            try:
                model.fit(series_train)
                
                future_dates = test_df.loc[series_id].index[:global_length]
                series_pred = model.predict(future_dates)
                
            except Exception as e:
                print(f"Error in Prophet prediction for series {series_id}: {str(e)}")
                # If Prophet fails, we'll skip this series for Prophet and fallback to a simple forecaster
                try:
                    simple_forecaster.fit(series_train)
                    series_pred = (simple_forecaster.predict(global_length)).values
                except Exception as e:
                    print(f"Error in Prophet's fallback - SimpleForecaster prediction for series {series_id}: {str(e)}. Using last known value.")
                    series_pred = np.full(global_length, series_train[target_col].iloc[-1])
            
            # Ensure predictions match the expected length
            if len(series_pred) < global_length:
                series_pred = np.pad(series_pred, (0, global_length - len(series_pred)), 'edge')
            elif len(series_pred) > global_length:
                series_pred = series_pred[:global_length]

            return np.array(series_pred)
        
        model_lib['Prophet'] = {'model': prophet, 'pred': prophet_pred}

    if 'GRU' in bench_models:
        gru = GRUForecaster(
                input_size=len(covariate_cols),
                hidden_size=hidden_size,
                output_size=global_length,
                num_layers=num_layers,
                epochs=epochs,
                batch_size=batch_size
            )
        
        model_lib['GRU'] = {'model': gru, 'pred': xy_pred}

    if 'TCN' in bench_models:
        tcn = TCNForecaster(
                input_size=len(covariate_cols),
                output_size=global_length,
                num_channels=[32, 32, 32, 32],
                kernel_size=3,
                dropout=0.2,
                epochs=epochs,
                batch_size=batch_size
            )
        model_lib['TCN'] = {'model': tcn, 'pred': xy_pred}
    
    if 'DeepAR local' in bench_models:
        deepar_local = LocalDeepARPredictor(
            input_size=len(covariate_cols + [weight_col]),
            hidden_size=hidden_size,
            num_layers=num_layers,
            input_length=local_length,
            output_length=global_length,
            dropout=0.2,
            epochs=epochs
        )

        # print("Training local DeepAR model...")
        deepar_local.fit(X_train, y_train, covariates_train)

        # print("Generating predictions...")
        deepar_local_predictions = deepar_local.predict(X_test, test_df[covariate_cols + [weight_col]], 
                                    steps=global_length)
        
        def local_deepar_pred(model, 
                        model_str, 
                        train_df, 
                        test_df, 
                        series_id, 
                        target_col, 
                        local_length, 
                        global_length,
                        predictions_df = deepar_local_predictions):
        
            series_pred = predictions_df[series_id][:global_length]

            # Ensure predictions match the expected length
            if len(series_pred) < global_length:
                series_pred = np.pad(series_pred, (0, global_length - len(series_pred)), 'edge')
            elif len(series_pred) > global_length:
                series_pred = series_pred[:global_length]

            return np.array(series_pred)
    
        model_lib['DeepAR local'] = {'model': deepar_local, 'pred': local_deepar_pred}

    if 'DeepAR global' in bench_models:
        deepar_global = GlobalDeepARPredictor(
            input_size=len(covariate_cols + [weight_col]),
            hidden_size=hidden_size,
            num_layers=num_layers,
            input_length=local_length,
            output_length=global_length,
            epochs=epochs
        )

        # print("Training global DeepAR model...")
        deepar_global.fit(X_train, y_train, covariates_train)

        # print("Generating predictions...")
        deepar_global_predictions = deepar_global.predict(
            X_test, 
            test_df[covariate_cols + [weight_col]], 
            steps=global_length
        )
        
        def global_deepar_pred(model, 
                        model_str, 
                        train_df, 
                        test_df, 
                        series_id, 
                        target_col, 
                        local_length, 
                        global_length,
                        predictions_df = deepar_global_predictions):
        
            series_pred = predictions_df[series_id][:global_length]

            # Ensure predictions match the expected length
            if len(series_pred) < global_length:
                series_pred = np.pad(series_pred, (0, global_length - len(series_pred)), 'edge')
            elif len(series_pred) > global_length:
                series_pred = series_pred[:global_length]

            return np.array(series_pred)
    
        model_lib['DeepAR global'] = {'model': deepar_global, 'pred': global_deepar_pred}

    # Prepare results storage
    local_results = {
        'T2f': {'mae': [], 'rmse': [], 'smape': [], 'wsmae': []}
    }

    global_results = {
        'T2f':  {'true': [], 'pred': [], 'weights': []}
    }

    for m in bench_models:
        local_results[m] = {'mae': [], 'rmse': [], 'smape': [], 'wsmae': []}
        global_results[m] =  {'true': [], 'pred': [], 'weights': []}

    # Evaluate models for each series
    print("\nStarting model comparison...")
    # print(f"Local input length: {local_local_length}")

    for series_id in series_ids:
        # T2f evaluation
        series_true = y_test.loc[series_id].values[:global_length]
        series_weights = weights_test.loc[series_id].values[:global_length]
        series_pred_t2f = t2f_predictions[t2f_predictions['series_id'] == series_id]['prediction'].values[:len(series_true)]
        #series_pred_t2f = (t2f_predictions[t2f_predictions.index.isin([series_id], level=0)])['prediction'].values[:global_length]
        
        local_metrics_t2f = evaluate_forecast(series_true, series_pred_t2f, series_weights)
        for metric, value in local_metrics_t2f.items():
            local_results['T2f'][metric].append(value)

        global_results['T2f']['true'].extend(series_true)
        global_results['T2f']['pred'].extend(series_pred_t2f)
        global_results['T2f']['weights'].extend(series_weights)

        ###########################################################################

        print(f'Benchmarking series {series_id}')
        for m in bench_models:
            print(f'Running {m}')
            model = model_lib[m]['model']
            predictor = model_lib[m]['pred']
            series_pred = predictor(model, m, train_df, test_df, series_id, target_col, local_length, global_length)
            series_pred = series_pred[:len(series_true)] ## ensure length match for evaluation

            try:
                local_metrics = evaluate_forecast(series_true, series_pred, series_weights)
                for metric, value in local_metrics.items():
                    local_results[m][metric].append(value)
            except:
                 # If it fails, skip series
                for metric in local_results[m]:
                    local_results[m][metric].append(np.nan)

            if m in local_results and series_pred is not None:  # Add None check
                try:
                    if isinstance(series_pred, (list, np.ndarray)):
                        global_results[m]['true'].extend(series_true)
                        global_results[m]['pred'].extend(series_pred)
                        global_results[m]['weights'].extend(series_weights)
                    else:
                        logger.warning(f"Model {m} returned invalid predictions for series {series_id}. Skipping.")
                        # Add NaN values to maintain alignment
                        global_results[m]['true'].extend([np.nan] * len(series_true))
                        global_results[m]['pred'].extend([np.nan] * len(series_true))
                        global_results[m]['weights'].extend([np.nan] * len(series_true))
                except Exception as e:
                    logger.error(f"Error storing results for model {m}, series {series_id}: {str(e)}")
                    # Add NaN values to maintain alignment
                    global_results[m]['true'].extend([np.nan] * len(series_true))
                    global_results[m]['pred'].extend([np.nan] * len(series_true))
                    global_results[m]['weights'].extend([np.nan] * len(series_true))

    return local_results, global_results