# sathvikswaminathan.github.io

Personal technical blog — notes on computer architecture, GPUs, performance engineering and systems software.

Live at **https://sathvikswaminathan.github.io**

## Stack

- [Astro](https://astro.build/) + [Tailwind CSS](https://tailwindcss.com/)
- [AstroPaper](https://github.com/satnaing/astro-paper) theme by Sat Naing (MIT — see `LICENSE`)
- Static search via [Pagefind](https://pagefind.app/)
- Deployed by GitHub Actions to GitHub Pages (`.github/workflows/deploy.yml`)

## Local development

Requires Node >= 22.12 and pnpm 10.

```bash
pnpm install
pnpm dev       # http://localhost:4321
pnpm build     # astro check + build + pagefind index
pnpm preview   # serve the production build
```

## Writing a post

Add a Markdown file to `src/content/posts/`. The filename becomes the slug.

```markdown
---
author: Sathvik Swaminathan
pubDatetime: 2026-10-20T05:00:00Z
title: Your Post Title
featured: false
tags:
  - gpu
description: One-line summary used for SEO and post listings.
---

Body goes here.
```

Required frontmatter: `title`, `description`, `pubDatetime`. Optional: `author`,
`modDatetime`, `featured`, `draft`, `tags`, `ogImage`, `canonicalURL`,
`hideEditPost`, `hideComments`.

> **Note:** `pubDatetime` must be in the past or the post is treated as scheduled
> and hidden from production builds. The 15-minute grace window is set by
> `posts.scheduledPostMargin` in `astro-paper.config.ts`.

## Configuration

Site title, description, socials, and feature flags live in `astro-paper.config.ts`.

## Comments and analytics

Both are client-side embeds configured under `features` in `astro-paper.config.ts`,
and both stay inert until their IDs are filled in — an empty value renders nothing
and logs a warning during `pnpm build`. The IDs are public identifiers that appear
in page source by design, so they belong in the repo, not in Actions secrets.

### Comments — [giscus](https://giscus.app) (GitHub Discussions)

Comments appear at the bottom of each post as a Discussion thread. Reply from the
page or from the repo's Discussions tab. Set `hideComments: true` in a post's
frontmatter to opt that post out.

One-time setup:

1. Repo **Settings → General → Features → Discussions**: enable.
2. In **Discussions**, create a category named `Comments` with type
   **Announcement**, so only you can open top-level threads.
3. Install the [giscus GitHub App](https://github.com/apps/giscus) on the repo.
4. On [giscus.app](https://giscus.app), enter the repo and pick the `Comments`
   category, then copy the generated `data-repo-id` and `data-category-id` into
   `features.comments.repoId` and `features.comments.categoryId`.

The widget follows the site's light/dark toggle; override the giscus themes with
`features.comments.lightTheme` / `darkTheme`.

### Analytics — [GoatCounter](https://www.goatcounter.com)

Cookie-free, so no consent banner is needed. Create a site on GoatCounter (free
for non-commercial use) and set `features.analytics.endpoint` to
`https://<your-code>.goatcounter.com/count`.

Because the site uses View Transitions, pageviews are counted manually on
`astro:page-load` rather than on document load. GoatCounter's own filter drops
localhost and private addresses, so `pnpm dev` traffic is never recorded.

## Deploying

Push to `main`. The workflow builds the site and publishes it to GitHub Pages.
