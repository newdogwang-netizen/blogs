// The Jekyll site is built with `baseurl: "/blogs"` (GitHub Pages convention).
// Cloudflare Workers serves static assets from the repository root, so the
// /blogs prefix is stripped before asset lookup. Without this, every
// sub-resource URL in the generated HTML (/blogs/...) would 404.
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/blogs") {
      url.pathname = "/blogs/";
    }
    if (url.pathname === "/blogs/" || url.pathname.startsWith("/blogs/")) {
      url.pathname = url.pathname.slice("/blogs".length) || "/";
      return env.ASSETS.fetch(new Request(url, request));
    }
    return env.ASSETS.fetch(request);
  },
};
