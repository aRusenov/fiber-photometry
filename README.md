# fiber-photometry

```
export set SCRIPT_DIR=""
export set DATA_DIR=""
export set OUT_DIR=""
```

## Artifact cleaning
```shell
python $SCRIPT_DIR/artifact_clean.py --channel AIN01 --dio DIO01 --file "$DATA_DIR/102_1_Det12_DIO1_0003.doric"
```

## Pre-processing
```shell
raw_files=$(find "$(cd .; pwd)" -name '*.doric' | grep -v 'R.doric')
echo $files
python $SCRIPT_DIR/preprocess.py --channel AIN01 --dio DIO01 DIO02 --label L --outdir "$OUT_DIR/preprocessed" --file $raw_files
```

## Plot
### Fat licking
```shell
python $SCRIPT_DIR/licking/licking_plot.py --channel AIN01 --dio DIO01 --label Left --outdir "$OUT_DIR/plots" --file $files
```

### 5C
```shell
python $SCRIPT_DIR/plot/5c_process_new.py --label Left --dio01 DIO01 --dio02 DIO02 --outdir "$OUT_DIR/plots" --file $OUT_DIR/preprocessed/656_3-L.h2py
python $SCRIPT_DIR/plot/5c_plot.py --outdir "$OUT_DIR/plots" --file $files
```


### Grep misc
Match regex
'find "$(cd .; pwd)" -name '*.h2py' | grep -Ei '/656_3-(R|L)''

Match negative
'find "$(cd .; pwd)" -name '*.h2py' | grep -v '/656_3-R'

Match simple
'find "$(cd .; pwd)" -name '*.h2py' | grep '/656_3-R'
