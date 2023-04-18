#! /bin/bash

BASEDIR=../DDI

# convert datasets to feature vectors
echo "Extracting features..."
python3 extract-features.py $BASEDIR/data/train/ > train.feat
python3 extract-features.py $BASEDIR/data/devel/ > devel.feat

# train CRF model
echo "Training CRF model..."
python3 train-crf.py model.crf < train.feat
# run CRF model
echo "Running CRF model..."
python3 predict.py model.crf < devel.feat > devel-CRF.out
# evaluate CRF results
echo "Evaluating CRF results..."
python3 evaluator.py NER $BASEDIR/data/devel devel-CRF.out > devel-CRF.stats

echo "FINISHED CRF"
#Extract Classification Features
cat train.feat | cut -f5- | grep -v ^$ > train.clf.feat


# train Naive Bayes model
echo "Training Naive Bayes model..."
python3 train-sklearn.py model.joblib vectorizer.joblib < train.clf.feat
# run Naive Bayes model
echo "Running Naive Bayes model..."
python3 predict-sklearn.py model.joblib vectorizer.joblib < devel.feat > devel-NB.out
# evaluate Naive Bayes results
echo "Evaluating Naive Bayes results..."
python3 evaluator.py NER $BASEDIR/data/devel devel-NB.out > devel-NB.stats


# train SVC model
echo "Training SVC model..."
python3 train-svc.py model.joblib vectorizer.joblib < train.clf.feat
# run SVC model
echo "Running SVC model..."
python3 predict-sklearn.py model.joblib vectorizer.joblib < devel.feat > devel-SVC.out
# evaluate Naive Bayes results
echo "Evaluating SVC results..."
python3 evaluator.py NER $BASEDIR/data/devel devel-SVC.out > devel-SVC.stats

# remove auxiliary files.
rm train.clf.feat
