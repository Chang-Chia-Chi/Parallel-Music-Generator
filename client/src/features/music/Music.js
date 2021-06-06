import React, { useState, useEffect } from "react";
// import { useSelector, useDispatch } from "react-redux";
import { IconButton, Typography } from "@material-ui/core";
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

function playMusic(msg) {
  let notes = msg;
  if (typeof notes === "string") {
    notes = JSON.parse(msg).music;
  } else {
    notes = msg.music || msg;
  }

  if (Array.isArray(notes)) {
    // notes = notes.slice(0, notes.length > 0 ? 1 : 0);
    // console.log("notes.length:", notes.length);

    // reset transport
    Tone.Transport.stop();
    Tone.Transport.cancel();
    Tone.Transport.seconds = 0;
    const synth = new Tone.Synth().toDestination();
    let currTime = Tone.now();

    // parse through the notes we are getting
    for (const mainNote of notes) {
      for (const subNote of mainNote) {
        const note =
          subNote[0] >= Config.NOTE_MAPPINGS.length
            ? subNote[0] % Config.NOTE_MAPPINGS.length
            : subNote[0];
        const duration =
          subNote[1] >= Config.NOTE_DURATIONS.length
            ? subNote[1] % Config.NOTE_DURATIONS.length
            : subNote[1];
        if (note === Config.REST_NOTE) {
          // rest
          Tone.Transport.scheduleOnce((time) => {
            synth.triggerAttackRelease(null, null, time);
          }, currTime);
        } else if (note < Config.REST_NOTE) {
          // note
          Tone.Transport.scheduleOnce((time) => {
            synth.triggerAttackRelease(
              Config.NOTE_MAPPINGS[note],
              Config.NOTE_DURATIONS[duration],
              time
            );
          }, currTime);
        } else {
          // chord
          let tempNotes = parseChord(note);
          for (const tempNote of tempNotes) {
            Tone.Transport.scheduleOnce((time) => {
              synth.triggerAttackRelease(
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
    // console.log("Start playing at", Tone.Transport.bpm.value);
    Tone.Transport.start();
  }
}

export function Music() {
  const classes = useStyles();

  // const dispatch = useDispatch();
  const [ws, setWs] = useState(null);
  if (!ws) {
    setWs(new WebSocket(Config.WS_URL));
  }

  useEffect(() => {
    if (ws) {
      //set websocket listener
      ws.onopen = () => {
        // on connecting, do nothing but log it to the console
        console.log("connected");
      };

      ws.onmessage = (evt) => {
        // on receiving a message, add it to the list of messages
        const message =
          evt.data && typeof evt.data === "string"
            ? evt.data
            : JSON.parse(evt.data);
        console.log("onmessage", message);

        playMusic(message);
      };

      ws.onclose = () => {
        console.log("disconnected");
        // automatically try to reconnect on connection loss
        setWs(new WebSocket(Config.WS_URL));
      };
    }
  }, [ws]);

  const sendMessage = (payload) => {
    //以 emit 送訊息，並以 getMessage 為名稱送給 server 捕捉
    console.log("sendMessage", payload);
    if (ws && ws.readyState !== WebSocket.CLOSED) {
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
        <IconButton
          color="primary"
          aria-label="happy"
          component="span"
          onClick={sendMessage.bind(this, { tune: 1 })}
        >
          <Mood className={classes.icon} />
        </IconButton>
        <IconButton
          color="secondary"
          aria-label="sad"
          component="span"
          onClick={sendMessage.bind(this, { tune: 2 })}
        >
          <MoodBad className={classes.icon} />
        </IconButton>
      </div>
    </div>
  );
}
