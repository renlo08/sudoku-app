document.addEventListener('DOMContentLoaded', function () {
  // Function to initialize the board elements
  function initializeBoardElements() {
    // Get the 'Frozen' and 'Edit' elements
    const frozenElements = document.querySelectorAll('.frozenBoardElt');
    const editElements = document.querySelectorAll('.editBoardElt');
    const editButton = document.querySelector('.editBoardButton');

    // Add event listener to the 'Edit' button
    if (editButton) {
      editButton.addEventListener('click', () => {
        toggleBoardElements();
        // Change the text of the editButton to 'Switch to Play Mode'
        if (editButton.textContent === 'Éditer') {
          editButton.textContent = 'Terminer';
        } else {
          editButton.textContent = 'Éditer';
        }
      });
    }

    // Function to toggle the display of elements
    function toggleDisplay(element) {
      if (element.style.display === 'none' || element.style.display === '') {
        element.style.display = 'block';
      } else {
        element.style.display = 'none';
      }
    }

    // Function to toggle the board elements
    function toggleBoardElements() {
      editElements.forEach(toggleDisplay);
      frozenElements.forEach(toggleDisplay);
    }
  }

  // Initialize board elements on page load
  initializeBoardElements();

  // Re-initialize board elements after each htmx request
  document.addEventListener('htmx:afterRequest', function () {
    initializeBoardElements();
  });
});