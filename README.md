# BLINK_AUTH
Codes for the paper titled, "Blink to Get In: Biometric Authentication forMobile Devices using EEG Signals"

## Citation
------------
The dataset and codes are freely available for research use. Please cite the following publication for using the codes and dataset
<blockquote>
  <p>Mohit Agarwal, Raghupathy Sivakumar<br />
BLINK: A Fully Automated Unsupervised Algorithm for Eye-Blink Detection in EEG Signals<br />
57th Annual Allerton Conference on Communication, Control, and Computing (Allerton). IEEE, 2019.</p>
</blockquote>

<blockquote>
  <p>Ekansh Gupta, Mohit Agarwal, Raghupathy Sivakumar<br />
Blink to Get In: Biometric Authentication forMobile Devices using EEG Signals<br />
IEEE International Conference on Communications. ICC, 2020.</p>
</blockquote>

## Dataset
----------
Dataset can be downloaded from [the link](http://gnan.ece.gatech.edu/eeg-eyeblinks/)

| Dataset       |  Type     | Users         | Blinks|
| ------------- |:----------|:-------------:| -----:|
| EEG-IO        |Involuntary| 20            | 500   |
| EEG-VV        |Voluntary  | 12            | 750   |
| EEG-VR        |Voluntary  | 12            | 600   |

## Codes
--------

Put `EEG-IO` or `EEG-VV` or `EEG-VR` dataset in `data/` folder, and choose the appropriate options in `blink_proc.py`.
Run `blink_proc.py` for replicating the results presented in the paper



## Contact
----------
[Mohit Agarwal](http://agmohit.com)(Email: me.agmohit@gmail.com)
Ekansh Gupta (egupta8@gatech.edu)

## References
-------------

[1] [Blink: A Fully Automated Unsupervised Algorithm for Eye Blink Detection in EEG Signals](http://gnan.ece.gatech.edu/archive/agarwal-blink.pdf)
