# fiber-photometry

## Artifact cleaning
```shell
python $SCRIPT_DIR/artifact_clean.py --channel AIN01 --dio DIO01 --file "/Users/atanas/Documents/workspace/data/B2 fat licking/102_1_Det12_DIO1_0003.doric"
```

## Pre-processing
```shell
files=$(find "$(cd .; pwd)" -name '*.doric' | grep -v 'R.doric')
echo $files
python $SCRIPT_DIR/preprocess.py --channel AIN01 --dio DIO01 --label L --outdir="/Users/atanas/Documents/workspace/data/B2 fat licking/preprocessed" --file $files
```

## Plot
```shell
python $SCRIPT_DIR/licking/licking_plot.py --channel AIN01 --dio DIO01 --label Left --outdir="/Users/atanas/Documents/workspace/data/B2 fat licking/plots/left.pdf" --file $files
```
