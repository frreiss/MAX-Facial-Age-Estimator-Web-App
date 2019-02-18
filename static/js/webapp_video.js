/*
 * Copyright 2018 IBM Corp. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

(function(exports) {

  ////////////////////////////////////////////////////////////////////////////////
  // Cross-platform webcam access boilerplate 
  navigator.getUserMedia = navigator.webkitGetUserMedia || navigator.msGetUserMedia 
    || navigator.getUserMedia || navigator.mozGetUserMedia;
  exports.requestAnimationFrame = exports.mozRequestAnimationFrame 
    || exports.webkitRequestAnimationFrame || exports.requestAnimationFrame 
    || exports.oRequestAnimationFrame || exports.msRequestAnimationFrame;
  exports.cancelAnimationFrame = exports.mozCancelAnimationFrame 
    || exports.cancelAnimationFrame || exports.msCancelAnimationFrame 
    || exports.webkitCancelAnimationFrame || exports.oCancelAnimationFrame;
  exports.URL = exports.URL || exports.webkitURL;

  ////////////////////////////////////////////////////////////////////////////////  
  // Constants
  const _VIDEO_WIDTH_PX = 1024 
  const _VIDEO_HEIGHT_PX = 576
  const _JPEG_COMPRESSION = 0.9
  const _TARGET_FPS = 15
  const _FRAME_INTERVAL_MSEC = 1000.0 / _TARGET_FPS

  // Parameters of the PID controller for frame rate
  const _DECAY_FACTOR = 0.9
  const _P = 0.6
  const _I = 5.0
  const _D = 0.01


  ////////////////////////////////////////////////////////////////////////////////
  // Global variables for the handlers below
  initEvents();
  exports.$ = $;
  var ORIGINAL_DOC_TITLE = document.title;
  var mycanvas = document.createElement('canvas');
  var video = $('video');
  var rafId = null;

  // Variables for PID controller
  var lastFrameMsec = -1
  var integralError = 0.0    // Exponential moving average
  var prevError = 0.0

  // Callback that sends video frames to backend
  var sendFrameCB = null;

  namespace = '/streaming';
  // console.log('http://' + document.domain + ':' + location.port + namespace);
  var socket = io.connect('http://' + document.domain + ':' + location.port + namespace);

  ////////////////////////////////////////////////////////////////////////////////
  // Subroutines
  
  /** 
   * Compute how long to sleep until it's time to send the next frame. 
   * Uses the following global variables to adjust the delay:
   *   -- lastFrameMsec (timestamp of last frame)
   *   -- integralError (integral of error, with exponential smoothing)
   *   -- prevError (error in delay of previous frame)
   */
  function msecToNextFrame() {
    curMsec = Date.now()
    if (lastFrameMsec < 0.0) {
        // Avoid weird time intervals on first call to this function.
        lastFrameMsec = curMsec - _FRAME_INTERVAL_MSEC;
    }
    msecSinceLastFrame = curMsec - lastFrameMsec;
    lastFrameMsec = curMsec

    // PID control; see https://en.wikipedia.org/wiki/PID_controller
    // Error is difference between target intra-frame delay and measured
    // delay.
    curError = _FRAME_INTERVAL_MSEC - msecSinceLastFrame
    integralError = (integralError * _DECAY_FACTOR) 
        + (curError * (1.0 - _DECAY_FACTOR))
    derivError = curError - prevError
    prevError = curError
    targetDelay = _FRAME_INTERVAL_MSEC + (_P * curError) + (_I * integralError)
                  - (_D * derivError)
    return Math.max(0.0, targetDelay);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Event handlers

  function $(selector) {
    return document.querySelector(selector) || null;
  }
  socket.on('connected', function () {
      socket.emit('netin', { data: 'Connected!' });
    });

  function initEvents() {
    $('#webcame').addEventListener('click', WebcamON);
  };


  /** Handler for the "start webcam" button. */
  function WebcamON(e) {
    video.height = _VIDEO_HEIGHT_PX;
    video.width = _VIDEO_WIDTH_PX;

    video.onloadedmetadata = function() {
      // console.log('in onloadedmetadata');
      video.play();
    };

    navigator.mediaDevices.getUserMedia({ audio: false, 
        video: { width: _VIDEO_WIDTH_PX, height: _VIDEO_HEIGHT_PX }})
      .then(function(stream) {
          // console.log('after getUserMedia');
          video.srcObject = stream;
          mycanvas.height = video.height;
          mycanvas.width = video.width;
        })
      .catch(function(err) {
          console.log('err ' + err);
        });

    var ctx = mycanvas.getContext('2d');
    socket.emit('netin', { data: 'Run Estimator!' });

    function sendVideoFrame_() {
      ctx.drawImage(video, 0, 0, mycanvas.width, mycanvas.height);
      socket.emit('streamingvideo', { data: mycanvas.toDataURL('image/jpeg', 
        _JPEG_COMPRESSION) });
      sendFrameCB = setTimeout(function(){sendVideoFrame_()}, msecToNextFrame());
    };

    // Use setTimeout(), not setInterval(), to avoid queueing events in the
    // browser.
    sendFrameCB = setTimeout(function(){sendVideoFrame_()}, 0.0);

    video.style.display="none";
  };
})(window);
