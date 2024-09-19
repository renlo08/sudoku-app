document.addEventListener('DOMContentLoaded', function () {
    const toggleButton = document.querySelector('.editBoard');
  
    // Hide all editBoardValue and show all frozenBoardValue when the page loads
    const editDropdowns = document.querySelectorAll('.editBoardValue');
    const frozenDropdowns = document.querySelectorAll('.frozenBoardValue');
    
    editDropdowns.forEach((dropdown) => {
      dropdown.style.display = 'none';
    });
    
    frozenDropdowns.forEach((dropdown) => {
      dropdown.style.display = 'block';
    });
  
    toggleButton.addEventListener('click', function () {
      editDropdowns.forEach((dropdown) => {
        if (dropdown.style.display === 'none' || dropdown.style.display === '') {
          dropdown.style.display = 'block';
        } else {
          dropdown.style.display = 'none';
        }
      });
      
      frozenDropdowns.forEach((dropdown) => {
        if (dropdown.style.display === 'none' || dropdown.style.display === '') {
          dropdown.style.display = 'block';
        } else {
          dropdown.style.display = 'none';
        }
      });
    });
  });