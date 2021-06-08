import React, { useState, useEffect } from "react";
// import { useSelector, useDispatch } from "react-redux";
import {
  IconButton,
  Typography,
  Backdrop,
  CircularProgress,
} from "@material-ui/core";
import { Mood, MoodBad } from "@material-ui/icons";
import { makeStyles } from "@material-ui/core/styles";
import * as Config from "../../constants/Config";
import * as Tone from "tone";

const useStyles = makeStyles({
  root: {
    margin: "auto",
  },
  icon: {
    width: "150px",
    height: "150px",
  },
  container: {
    display: "flex",
    flexDirection: "row",
  },
});

const synth = new Tone.Synth().toDestination();

const piano = new Tone.Sampler(
  Config.PIANO_SETTINGS_FILES,
  Config.PIANO_SETTINGS_URL
).toDestination();
piano.volume.value = -18;

const violin = new Tone.Sampler(
  Config.VIOLIN_SETTINGS_FILES,
  Config.VIOLIN_SETTINGS_URL
).toDestination();
violin.volume.value = 12;

const xylo = new Tone.Sampler(
  Config.XYLO_SETTINGS_FILES,
  Config.XYLO_SETTINGS_URL
).toDestination();
xylo.volume.value = 12;

const trumpet = new Tone.Sampler(
  Config.TRUMPET_SETTINGS_FILES,
  Config.TRUMPET_SETTINGS_URL
).toDestination();
trumpet.volume.value = 12;

const tuba = new Tone.Sampler(
  Config.TUBA_SETTINGS_FILES,
  Config.TUBA_SETTINGS_URL
).toDestination();
tuba.volume.value = 12;

const SYNTHS = [
  [piano, piano, piano],
  [piano, synth, synth],
  [piano, synth, synth],
  [violin, synth, synth],
  [xylo, synth, synth],
  [xylo, synth, synth],
  [trumpet, synth, synth],
  [tuba, tuba, tuba],
  [synth, synth, synth],
  [
    new Tone.Synth({
      oscillator: {
        type: "sine",
        volume: Config.VOLUME,
      },
      envelope: {
        attack: Config.ADSR.A,
        decay: Config.ADSR.D,
        sustain: Config.ADSR.S,
        release: Config.ADSR.R,
      },
    }).toDestination(),
    synth,
    synth,
  ],
];

var lpf = new Tone.Filter({
  type: "lowpass",
  frequency: Config.LOWPASS_FREQ,
}).toDestination();

var hpf = new Tone.Filter({
  type: "highpass",
  frequency: Config.HIGHPASS_FREQ,
}).connect(lpf);

// connect to speakers
SYNTHS.forEach((s) => {
  s[0].connect(hpf);
  s[1].connect(hpf);
  s[2].connect(hpf);
});

function parseChord(chord) {
  // Convert to base 12
  let temp = chord - Config.CHORD_OFFSET;
  temp = temp.toString(Config.DISTINCT_NOTES);

  let notes = [];

  // first iteration
  let prevNote =
    Config.NOTE_TYPES[parseInt(temp[temp.length - 1], Config.DISTINCT_NOTES)];
  let octave = Config.CHORD_START;

  notes.push(prevNote + octave);

  for (let i = temp.length - 2; i >= 0; i--) {
    if (parseInt(temp[i]) < prevNote) {
      octave++;
    }

    prevNote = Config.NOTE_TYPES[parseInt(temp[i], Config.DISTINCT_NOTES)];

    notes.push(prevNote + octave);
  }

  return notes;
}

function playMusic(msg, musicBpm) {
  let notes = msg;
  if (typeof notes === "string") {
    notes = JSON.parse(msg).music;
  } else {
    notes = msg.music || msg;
  }

  if (Array.isArray(notes)) {
    // notes = notes.slice(0, notes.length > 0 ? 1 : 0);
    // console.log("play music");

    // reset transport
    Tone.context.resume();
    Tone.Transport.stop();
    Tone.Transport.cancel();
    Tone.Transport.seconds = 0;
    console.log('musicBpm:', musicBpm);
    Tone.Transport.bpm.value = musicBpm;
    Tone.loaded().then(() => {
      console.log("Tone loaded!");
      let currTime = 0;
      // parse through the notes we are getting
      for (const [mi, mainNote] of notes.entries()) {
        console.log('mi', mi)
        for (const subNote of mainNote) {
          const note = subNote[0];
          const duration =
            subNote[1] >= Config.NOTE_DURATIONS.length
              ? subNote[1] % Config.NOTE_DURATIONS.length
              : subNote[1];
          if (note === Config.REST_NOTE) {
            // rest
            Tone.Transport.scheduleOnce((time) => {
              console.log("scheduleOnce rest", time, mi, 0);
              SYNTHS[mi][0].triggerAttackRelease(time);
            }, currTime);
          } else if (note < Config.REST_NOTE) {
            // note
            Tone.Transport.scheduleOnce((time) => {
              console.log("scheduleOnce < rest", time, mi, 0);
              SYNTHS[mi][0].triggerAttackRelease(
                Config.NOTE_MAPPINGS[note],
                Config.NOTE_DURATIONS[duration],
                time
              );
            }, currTime);
          } else {
            // chord
            let tempNotes = parseChord(note);
            for (const [ti, tempNote] of tempNotes.entries()) {
              Tone.Transport.scheduleOnce((time) => {
                console.log("scheduleOnce chord", time, mi, ti);
                SYNTHS[mi][ti].triggerAttackRelease(
                  tempNote,
                  Config.NOTE_DURATIONS[duration],
                  time
                );
              }, currTime);
            }
          }

          currTime += Tone.Time(Config.NOTE_DURATIONS[duration]).toSeconds();
        }
      }

      Tone.Transport.start();
    });
  }
}

export function Music() {
  const classes = useStyles();
  const [ws, setWs] = useState(null);

  const [btnDisabled, setBtnDisabled] = useState(true);
  const [openBackDrop, setBackDropOpen] = useState(false);
  const [musicBpm, setMusicBpm] = useState(120);

  if (!ws) {
    setWs(new WebSocket(Config.WS_URL));
  }

  useEffect(() => {
    if (ws) {
      //set websocket listener
      ws.onopen = () => {
        // on connecting, do nothing but log it to the console
        console.log("connected");
        setBtnDisabled(false);
      };

      ws.onmessage = (evt) => {
        // on receiving a message, add it to the list of messages
        const message =
          evt.data && typeof evt.data === "string"
            ? evt.data
            : JSON.parse(evt.data);
        console.log("onmessage", message);
        setBackDropOpen(false);
        setBtnDisabled(false);
        playMusic(message, musicBpm);
      };

      ws.onclose = () => {
        console.log("disconnected");
        setBtnDisabled(true);
        // automatically try to reconnect on connection loss
        setWs(new WebSocket(Config.WS_URL));
      };
    }
  }, [ws, musicBpm]);

  const sendMessage = (payload, bpm) => {
    console.log("sendMessage", payload);
    if (ws && ws.readyState !== WebSocket.CLOSED) {
      setBackDropOpen(true);
      setBtnDisabled(true);
      setMusicBpm(bpm);

      ws.send(
        JSON.stringify({
          action: "relay",
          payload,
        })
      );
    }
  };

  return (
    <div className={classes.root}>
      <Typography variant="h5" color="primary" gutterBottom>
        Click mood to play music !
      </Typography>
      <div className={classes.container}>
        <Backdrop className={classes.backdrop} open={openBackDrop}>
          <CircularProgress color="inherit" />
        </Backdrop>
        <IconButton
          color="primary"
          aria-label="happy"
          component="span"
          disabled={btnDisabled}
          onClick={sendMessage.bind(this, { tune: 2 }, 30)}
        >
          <Mood className={classes.icon} />
        </IconButton>
        <IconButton
          color="secondary"
          aria-label="sad"
          component="span"
          disabled={btnDisabled}
          onClick={sendMessage.bind(this, { tune: 1 }, 4)}
        >
          <MoodBad className={classes.icon} />
        </IconButton>
      </div>
    </div>
  );
}
