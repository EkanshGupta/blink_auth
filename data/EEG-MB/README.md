# [Dataset] FLUTTER-MultipleBlinks-OpenBCI
EEG dataset recording for 19 subjects containing multiple eye-blinks

## Citing the Dataset
----------------------------

The dataset is freely available for research use. Please cite the following publication is using
<blockquote>
  <p>Mohit Agarwal, Raghupathy Sivakumar<br />
Don’t charge for a day : Low-Power Wakeup Command Detection for Always-on BCI Wear-ables<br />
(Submitted)</p>
</blockquote>

## Dataset Specifications
----------------------------
*  **Sampling Frequency:** 250.0 samples per second
*  **Electrodes:** Fp1 and Fp2
*  **Total Subjects:** 19
*  **Experiment:** Subjects were asked to perform 3-blinks followed by 3 blinks (i.e. 3-3) and subjects were asked to restrict the movements during the experiment recording.

## Dataset Files
----------------
Raw EEG data is stored as `S<Sub_ID>_data.csv` and corresponding labels are stored in `S<Sub_ID>_labels.csv`

## Reading Data
----------------

A script `read_data.py` is provided to read the dataset 

#### Raw EEG data
Data is stored in a .csv format where column 0 represents time, and column 1 and 2 represents raw EEG potentials (in uV) for channel 1 (Fp1) and channel 2 (Fp2) respectively.

#### Labels
* The first line is `corrupt, <n>` where n represents the total number of corrupt intervals
* n following lines represent the start and end time of corrupt interval in seconds. A value of -1 means until the end.
* The next line is 'blinks' which marks the starting of blinks
* Blinks are arranged as `<blink_time>, <code>` where
  *  `<blink_time>`: middle point of blink in seconds (where the minima appears in EEG)
  * `<code>`: `0` is normal blink, `1` is blink when stimulation was given, `2` is soft blink


## Contact
----------------

For any queries, contact Mohit Agarwal
Email: me.agmohit@gmail.com
