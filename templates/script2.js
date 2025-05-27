document.addEventListener("DOMContentLoaded", () => {
  const buttons = document.querySelectorAll(".play-btn");
  let currentAudio = null;
  let currentButton = null;

  buttons.forEach(button => {
      button.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();

          const fileName = button.getAttribute("data-audio");
          const audioPath = `${fileName}`;

          // If this button is already playing audio, toggle stop/play
          if (currentButton === button && currentAudio && !currentAudio.paused) {
              // Stop the audio
              currentAudio.pause();
              currentAudio.currentTime = 0;
              button.textContent = "▶"; // Change to play icon
              currentAudio = null;
              currentButton = null;
              console.log(`⏹ Stopped: ${audioPath}`);
              return;
          }

          // Stop any currently playing audio
          if (currentAudio && !currentAudio.paused) {
              currentAudio.pause();
              currentAudio.currentTime = 0;
              if (currentButton) {
                  currentButton.textContent = "▶"; // Reset previous button to play icon
              }
          }

          // Create new audio and play it
          currentAudio = new Audio(audioPath);
          currentButton = button;

          currentAudio.play().then(() => {
              button.textContent = "⏹"; // Change to stop icon
              console.log(`🔊 Playing: ${audioPath}`);
              // Reset button when audio ends naturally
              currentAudio.addEventListener("ended", () => {
                  button.textContent = "▶";
                  currentAudio = null;
                  currentButton = null;
              }, { once: true });
          }).catch(err => {
              console.error("⚠️ Playback error:", err);
              alert(`Error playing audio: ${err.message}`);
              button.textContent = "▶"; // Reset on error
              currentAudio = null;
              currentButton = null;
          });
      });
  });
});