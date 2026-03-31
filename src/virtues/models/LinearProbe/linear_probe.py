import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from loguru import logger


def run_linear_probe(train_samples,
                    train_labels, 
                    test_samples, 
                    test_labels,
                    filter_labels=[],
                    id_to_name_dict=None,
                    balance_classes=False,
                    sampler=None,
                    sample_test_set=True,
                    return_confusion_matrix=False,
                    ):
    
    train_filter = (~np.isin(train_labels, filter_labels))
    test_filter = (~np.isin(test_labels, filter_labels))

    train_samples = train_samples[train_filter]
    train_labels = train_labels[train_filter]

    test_samples = test_samples[test_filter]
    test_labels = test_labels[test_filter]

    logger.info(f'Train samples shape: {train_samples.shape}')
    logger.info(f'Test samples shape: {test_samples.shape}')

    if sampler is not None:
        train_samples, train_labels = sampler.fit_resample(train_samples, train_labels)

        logger.info(f'Train samples shape after resampling: {train_samples.shape}')
        

        if sample_test_set:
            test_samples, test_labels = sampler.fit_resample(test_samples, test_labels)

    logger.info('Fitting model...')
    model = LogisticRegression(max_iter=1000, class_weight='balanced') if balance_classes else LogisticRegression(max_iter=1000)
    model.fit(train_samples, train_labels)

    classes = np.unique(np.concatenate([train_labels, test_labels]))
    decoded_classes = None
    if id_to_name_dict is not None:
        decoded_classes = [id_to_name_dict[x] for x in classes]

    logger.info('Decoded classes:', decoded_classes)

    train_report = classification_report(train_labels, model.predict(train_samples), labels=classes, target_names=decoded_classes, output_dict=True)
    test_report = classification_report(test_labels, model.predict(test_samples), labels=classes, target_names=decoded_classes, output_dict=True)

    logger.info("--------Train report--------")
    logger.info(classification_report(train_labels, model.predict(train_samples), labels=classes, target_names=decoded_classes))
    logger.info("--------Test report--------")
    logger.info(classification_report(test_labels, model.predict(test_samples), labels=classes, target_names=decoded_classes))

    if not return_confusion_matrix:
        return model, train_report, test_report
    test_confusion_matrix = confusion_matrix(test_labels, model.predict(test_samples), labels=classes)

    return model, train_report, test_report, test_confusion_matrix

def run_probe(
            model,
            train_samples,
            train_labels, 
            test_samples, 
            test_labels,
            filter_labels=[],
            sampler=None,
            sample_test_set=True,
            ):
    
    train_filter = (~np.isin(train_labels, filter_labels))
    test_filter = (~np.isin(test_labels, filter_labels))

    train_samples = train_samples[train_filter]
    train_labels = train_labels[train_filter]

    test_samples = test_samples[test_filter]
    test_labels = test_labels[test_filter]

    if sampler is not None:
        train_samples, train_labels = sampler.fit_resample(train_samples, train_labels)
        if sample_test_set:
            test_samples, test_labels = sampler.fit_resample(test_samples, test_labels)

    logger.info('Fitting model...')
    model.fit(train_samples, train_labels)

    train_report = classification_report(train_labels, model.predict(train_samples), output_dict=True)
    test_report = classification_report(test_labels, model.predict(test_samples), output_dict=True)

    logger.info("--------Train report--------")
    logger.info(classification_report(train_labels, model.predict(train_samples)))
    logger.info("--------Test report--------")
    logger.info(classification_report(test_labels, model.predict(test_samples)))

    return model, train_report, test_report








