// function related to selecting dynamic tab element
function activateTab(e) {
    var tabs = document.querySelectorAll('.tab');
    [].forEach.call(tabs, function(tab) {
        tab.classList.remove('tab-active');
    });
    e.target.classList.add('tab-active');
}

// function to the drag and drop feature
function dragOverDropZone(e) {
    e.preventDefault();
    e.target.classList.add('drag-over');
}

function dragLeaveDropZone(e) {
    e.preventDefault();
    e.target.classList.remove('drag-over');
}

function dropOnDropZone(e) {
    e.preventDefault();
    var dropZone = document.getElementById('drop-zone');
    dropZone.classList.remove('drag-over');
    var files = e.dataTransfer.files;
    if (files.length) {
        let fileInputElement = document.getElementById('new-file');
        fileInputElement.files = files;
        fileInputElement.dispatchEvent(new Event('change'));
    }
}