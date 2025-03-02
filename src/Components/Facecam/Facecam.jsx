

// Facecam.jsx
import React, { useState, useRef, useEffect } from 'react';
import './facecam.css';

const Facecam = () => {
  const [cameraOn, setCameraOn] = useState(false);
  const [recording, setRecording] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [questions, setQuestions] = useState([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);

  // Simulate fetching questions from backend
  useEffect(() => {
    // Replace this with actual API call
    const mockFetchQuestions = async () => {
      const sampleQuestions = [
        "Tell me about yourself",
        "What are your greatest strengths?",
        "Describe a challenging situation you faced and how you handled it",
        "Where do you see yourself in 5 years?",
        "Why do you want to work for our company?",
        "What is your greatest professional achievement?"
      ];
      setQuestions(sampleQuestions);
    };

    mockFetchQuestions();
  }, []);

  // Camera control
  const toggleCamera = async () => {
    try {
      if (!cameraOn) {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 1280, height: 720 },
          audio: true
        });
        mediaStreamRef.current = stream;
        videoRef.current.srcObject = stream;
        setCameraOn(true);
      } else {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
        videoRef.current.srcObject = null;
        setCameraOn(false);
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      alert('Failed to access camera. Please check permissions.');
    }
  };

  // Mute control
  const toggleMute = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getAudioTracks().forEach(track => {
        track.enabled = !track.enabled;
      });
      setIsMuted(!isMuted);
    }
  };

  // Question navigation
  const handleSkipQuestion = () => {
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(prev => prev + 1);
    }
  };

  // Cleanup
  useEffect(() => {
    return () => {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="facecam-page">
      <div className="facecam-container">
        <h1 className="p1">AI Interview Session</h1>
        
        <div className="facecam-video-container">
          <video 
            ref={videoRef}
            className="facecam-video" 
            autoPlay 
            playsInline 
            muted={isMuted}
          />
          {!cameraOn && (
            <div className="facecam-placeholder">Camera Preview</div>
          )}
        </div>

        <div className="facecam-controls">
          <button className="facecam-button" onClick={toggleCamera}>
            {cameraOn ? 'Stop Camera' : 'Start Camera'}
          </button>
          
          <button 
            className="facecam-button" 
            onClick={toggleMute}
            disabled={!cameraOn}
          >
            {isMuted ? 'Unmute' : 'Mute'}
          </button>
          
          <button 
            className="facecam-button"
            onClick={() => setRecording(!recording)}
            disabled={!cameraOn}
          >
            {recording ? 'Stop Recording' : 'Start Recording'}
          </button>
          <button className="facecam-button btn3">
            Leave Interview
           </button>
        </div>

        <div className="interview">
          <h3 className="p3">Interview Questions</h3>
          <div className="question-container">
            {questions.length > 0 ? (
              <>
                <div className="current-question">
                  Q{currentQuestionIndex + 1}: {questions[currentQuestionIndex]}
                </div>
                <button 
                  className="facecam-button skip-button"
                  onClick={handleSkipQuestion}
                  disabled={currentQuestionIndex >= questions.length - 1}
                >
                  Skip Question
                </button>
              </>
            ) : (
              <div>Loading questions...</div>
            )}
          </div>
          <button className='btn2'>Submit</button>
        </div>
      </div>
    </div>
  );
};

export default Facecam;
