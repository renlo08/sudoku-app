document.addEventListener('DOMContentLoaded', function () {
  // Get the 'Frozen' and 'Edit' elements
  const frozenElements = document.querySelectorAll('.frozenBoardElt');
  const editElements = document.querySelectorAll('.editBoardElt');
  const toggleButtons = document.querySelectorAll('.boardToggleButton');

  // Initially hide all 'Edit' elements
  editElements.forEach((element) => {
    element.style.display = 'none';
  });
  // Initially display all 'Frozen' elements
  frozenElements.forEach((element) => {
    element.style.display = 'block';
  });

  // Add event listeners to each 'Toggle' button
  toggleButtons.forEach((button) => {
    button.addEventListener('click', () => {
      toggleBoardElements();
    });
  });

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
    frozenElements.forEach((element) => {
      toggleDisplay(element);
    });
    editElements.forEach((element) => {
      toggleDisplay(element);
    });
  }
});