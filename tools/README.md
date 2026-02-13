Install [fusion2urdf](https://github.com/syuntoku14/fusion2urdf) by syuntoku14 in Fusion. Zipped file already available beside the README file.
Then use the xacro to urdf conversion tool (courtesy [xacro2urdf](https://github.com/doctorsrn/xacro2urdf) by doctorsrn), the needed file is already available as `xacro.py`

```
cd assets/Hiwonder_description && uv run ../../tools/xacro.py -o Hiwonder.urdf urdf/Hiwonder.xacro
```

