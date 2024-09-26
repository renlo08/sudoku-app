function initializeEditButton() {
  // Find all buttons with the class 'editBoardButton' to avoid missing any new elements after the swap
  let editButtons = document.querySelectorAll('.editBoardButton');
  
  // Loop over all buttons found and attach click listeners
  editButtons.forEach(editButton => {
    // Remove any existing event listener to avoid duplication
    editButton.removeEventListener('click', handleEditButtonClick);
    
    // Attach new event listener
    editButton.addEventListener('click', handleEditButtonClick);
  });
}

// Handle button click action to avoid duplication of inline function creation
function handleEditButtonClick(event) {
  const editButton = event.currentTarget;
  toggleBoardElements();
  toggleButtonName(editButton);
}

// Initialize board elements on page load
document.addEventListener('DOMContentLoaded', function () {
  initializeEditButton();
});

// Re-initialize board elements after each htmx swap
document.body.addEventListener('htmx:afterSwap', function (event) {
  // Ensure the swapped content contains the button (htmx:afterSwap is fired after DOM updates)
  if (event.detail.target) {
    initializeEditButton();
  }
});

// Function to toggle the name of the button
function toggleButtonName(button) {
  if (button.textContent === 'Éditer') {
    button.textContent = 'Terminer';
  } else {
    button.textContent = 'Éditer';
  }
}

// Function to toggle the display of board elements
function toggleBoardElements() {
  let frozenElements = document.querySelectorAll('.frozenBoardElt');
  let editElements = document.querySelectorAll('.editBoardElt');
  editElements.forEach(element => {
    element.style.display = element.style.display === 'none' ? '' : 'none';
  });
  frozenElements.forEach(element => {
    element.style.display = element.style.display === 'none' ? '' : 'none';
  });
}
