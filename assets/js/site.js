(function () {
  var root = document.documentElement;
  var toggle = document.querySelector('.theme-toggle');
  var progress = document.querySelector('.reading-progress span');
  var toc = document.querySelector('.article-toc');
  if (toc) {
    var list = toc.querySelector('ol');
    document.querySelectorAll('.post-body h2[id]').forEach(function (heading) {
      var item = document.createElement('li');
      var link = document.createElement('a');
      link.href = '#' + heading.id;
      link.textContent = heading.textContent;
      item.appendChild(link);
      list.appendChild(item);
    });
    toc.hidden = !list.children.length;
  }

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
