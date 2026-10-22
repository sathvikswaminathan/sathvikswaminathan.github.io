import { defineAstroPaperConfig } from "./src/types/config";

export default defineAstroPaperConfig({
  site: {
    url: "https://sathvikswaminathan.github.io/",
    title: "Sathvik Swaminathan",
    description:
      "Sathvik Swaminathan, Silicon Design Engineer at AMD. Learning GPUs from a CPU background.",
    author: "Sathvik Swaminathan",
    profile: "https://www.linkedin.com/in/sathvik-swaminathan-50b424178/",
    ogImage: "default-og.jpg",
    lang: "en",
    timezone: "America/Chicago",
    dir: "ltr",
  },
  posts: {
    perPage: 6,
    perIndex: 4,
    scheduledPostMargin: 15 * 60 * 1000,
  },
  features: {
    lightAndDarkMode: true,
    dynamicOgImage: true,
    showArchives: true,
    showBackButton: true,
    editPost: {
      enabled: true,
      url: "https://github.com/sathvikswaminathan/sathvikswaminathan.github.io/edit/main/",
    },
    search: "pagefind",
    // Comments via GitHub Discussions. Enable Discussions on the repo, add a
    // category named "Comments" (type: Announcement, so only you can open
    // top-level threads), install https://github.com/apps/giscus, then paste
    // the two IDs generated at https://giscus.app. Both are public, not secrets.
    // Comments stay hidden until both IDs are filled in.
    comments: {
      provider: "giscus",
      repo: "sathvikswaminathan/sathvikswaminathan.github.io",
      repoId: "R_kgDOVQcIQw",
      category: "Announcements",
      categoryId: "DIC_kwDOVQcIQ84DJVE6",
      // Chosen to match src/styles/theme.css. Light mode drops giscus's outer
      // border to sit flush on --background #fdfdfd; dark mode goes transparent
      // so it inherits --background #212737 exactly (giscus's own "dark" is
      // #0d1117, far darker than this site's navy-slate).
      lightTheme: "noborder_light",
      darkTheme: "transparent_dark",
    },
    // Cookie-free analytics. Create a site at https://www.goatcounter.com and
    // set the endpoint to https://<your-code>.goatcounter.com/count.
    // Analytics stays off until the endpoint is filled in.
    analytics: {
      provider: "goatcounter",
      endpoint: "https://sathvik.goatcounter.com/count",
    },
  },
  socials: [
    {
      name: "linkedin",
      url: "https://www.linkedin.com/in/sathvik-swaminathan-50b424178/",
    },
  ],
  shareLinks: [
    { name: "x", url: "https://x.com/intent/post?url=" },
    { name: "telegram", url: "https://t.me/share/url?url=" },
    { name: "mail", url: "mailto:?subject=See%20this%20post&body=" },
  ],
});
