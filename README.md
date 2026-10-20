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
`modDatetime`, `featured`, `draft`, `tags`, `ogImage`, `canonicalURL`.

> **Note:** `pubDatetime` must be in the past or the post is treated as scheduled
> and hidden from production builds. The 15-minute grace window is set by
> `posts.scheduledPostMargin` in `astro-paper.config.ts`.

## Configuration

Site title, description, socials, and feature flags live in `astro-paper.config.ts`.

## Deploying

Push to `main`. The workflow builds the site and publishes it to GitHub Pages.
