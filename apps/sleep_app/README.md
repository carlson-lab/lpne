## Semi-Automated Sleep Labeling App

This app assigns sleep labels (Wake, NREM, REM) given hippocampal and shoulder
EMG recordings.

### Running Locally
To run locally, navigate to the repo and enter the `bokeh serve` command:

```bash
$ cd path/to/lpne
$ bokeh serve apps/sleep_app/ --show
```

### Running Remotely
To run remotely, first run `bokeh serve` on the remote machine and specify a
port:

```bash
$ cd path/to/lpne
$ bokeh serve apps/sleep_app/ port=1234
```

This will spit out an address that looks something like this:
`http://localhost:1234/sleep_app`. Copy this address.

Then on your local machine, start listening to this port with `ssh`:

```bash
$ ssh -NL localhost:1234:localhost:1234 user@computer
```

where `user` is replaced by your username and `computer` is replaced by the
remote machine's name or IP address.

Lastly, on your local machine, open a browser enter go the address you copied.
