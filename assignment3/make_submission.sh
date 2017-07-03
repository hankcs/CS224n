#!/bin/bash
echo "Creating submission file."
PYFILES=`ls *.py`
CONLL_FILES="window_predictions.conll rnn_predictions.conll gru_predictions.conll"

CONLL=
for conll in $CONLL_FILES; do
    if [ -e $conll ]; then
        CONLL="$CONLL $conll"
    else
        echo "WARNING: Could not find $conll in this directory. If you
        have generated it, please move it from the appropriate folder in
        results/"
    fi
done;

zip submissions.zip $PYFILES $CONLL
echo "Done."
