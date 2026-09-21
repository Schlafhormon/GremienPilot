import { useRef, useState, useEffect, useCallback } from 'react';

interface AudioPlayerProps {
  audioUrl: string;
  currentTime?: number;
  onTimeUpdate?: (time: number) => void;
  playOnSeek?: boolean;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export default function AudioPlayer({
  audioUrl,
  currentTime,
  onTimeUpdate,
  playOnSeek = false,
}: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [internalTime, setInternalTime] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);

  // Handle external seek requests (e.g., when user clicks a transcript line)
  useEffect(() => {
    if (currentTime !== undefined && audioRef.current) {
      audioRef.current.currentTime = currentTime;
      if (playOnSeek) {
        void audioRef.current.play();
      }
    }
  }, [currentTime, playOnSeek]);

  const handleTimeUpdate = useCallback(() => {
    if (audioRef.current) {
      setInternalTime(audioRef.current.currentTime);
      onTimeUpdate?.(audioRef.current.currentTime);
    }
  }, [onTimeUpdate]);

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
    }
  };

  const togglePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleSeekBar = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newTime = parseFloat(e.target.value);
    if (audioRef.current) {
      audioRef.current.currentTime = newTime;
      setInternalTime(newTime);
    }
  };

  const skipBackward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.max(0, audioRef.current.currentTime - 10);
    }
  };

  const skipForward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.min(duration, audioRef.current.currentTime + 10);
    }
  };

  const changePlaybackRate = () => {
    const rates = [0.5, 0.75, 1, 1.25, 1.5, 2] as const;
    const currentIndex = rates.indexOf(playbackRate as typeof rates[number]);
    const nextIndex = (currentIndex + 1) % rates.length;
    const nextRate = rates[nextIndex] ?? 1;
    setPlaybackRate(nextRate);
    if (audioRef.current) {
      audioRef.current.playbackRate = nextRate;
    }
  };

  return (
    <div className="bg-gray-100 rounded-lg p-3 flex flex-wrap items-center gap-2">
      <audio
        ref={audioRef}
        src={audioUrl}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => setIsPlaying(false)}
        preload="metadata"
      />

      {/* Skip backward button */}
      <button
        onClick={skipBackward}
        className="p-2 hover:bg-gray-200 rounded-full text-gray-600"
        title="10 Sekunden zurueck"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
          <path d="M8.445 14.832A1 1 0 0010 14v-2.798l5.445 3.63A1 1 0 0017 14V6a1 1 0 00-1.555-.832L10 8.798V6a1 1 0 00-1.555-.832l-6 4a1 1 0 000 1.664l6 4z" />
        </svg>
      </button>

      {/* Play/Pause button */}
      <button
        onClick={togglePlayPause}
        aria-label={isPlaying ? 'Pause' : 'Abspielen'}
        className="p-2 bg-blue-600 text-white rounded-full hover:bg-blue-700 w-10 h-10 shrink-0 flex items-center justify-center"
      >
        {isPlaying ? (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
        ) : (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
          </svg>
        )}
      </button>

      {/* Skip forward button */}
      <button
        onClick={skipForward}
        className="p-2 hover:bg-gray-200 rounded-full text-gray-600"
        title="10 Sekunden vor"
      >
        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
          <path d="M4.555 5.168A1 1 0 003 6v8a1 1 0 001.555.832L10 11.202V14a1 1 0 001.555.832l6-4a1 1 0 000-1.664l-6-4A1 1 0 0010 6v2.798l-5.445-3.63z" />
        </svg>
      </button>

      {/* Time display */}
      <span className="text-sm text-gray-600 w-24 text-center">
        {formatTime(internalTime)} / {formatTime(duration)}
      </span>

      {/* Progress bar */}
      <input
        type="range"
        min={0}
        max={duration || 0}
        value={internalTime}
        onChange={handleSeekBar}
        aria-label="Audioposition"
        className="min-w-24 flex-1 h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer accent-blue-600"
      />

      {/* Playback rate button */}
      <button
        onClick={changePlaybackRate}
        className="px-2 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300 min-w-[3rem]"
        title="Wiedergabegeschwindigkeit"
      >
        {playbackRate}x
      </button>
    </div>
  );
}
