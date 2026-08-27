(function () {
  var root = document.documentElement;
  var toggle = document.querySelector('.theme-toggle');
  var progress = document.querySelector('.reading-progress span');

  if (toggle) {
    toggle.addEventListener('click', function () {
      var next = root.dataset.theme === 'dark' ? 'light' : 'dark';
      root.dataset.theme = next;
      localStorage.setItem('theme', next);
    });
  }

  if (progress && document.querySelector('.post-body')) {
    var updateProgress = function () {
      var height = document.documentElement.scrollHeight - window.innerHeight;
      var ratio = height > 0 ? window.scrollY / height : 0;
      progress.style.transform = 'scaleX(' + Math.min(1, Math.max(0, ratio)) + ')';
    };
    updateProgress();
    window.addEventListener('scroll', updateProgress, { passive: true });
  }
})();

