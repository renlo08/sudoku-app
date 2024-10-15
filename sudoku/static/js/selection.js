function activateTab(e) {
    var tabs = document.querySelectorAll('.tab');
    [].forEach.call(tabs, function(tab) {
        tab.classList.remove('tab-active');
    });
    e.target.classList.add('tab-active');
}
