interface SiteConfig {
  /** Deployed URL of the site, e.g. "https://example.com" */
  url: string;
  /** Blog title shown in header and meta tags */
  title: string;
  /** Short description used in SEO meta and RSS feed */
  description: string;
  /** Default post author name */
  author: string;
  /** Author profile URL (used in structured data) */
  profile?: string;
  /** Fallback OG image filename in /public, e.g. "og.jpg" */
  ogImage?: string;
  /** HTML lang attribute, defaults to "en" */
  lang?: string;
  /** IANA timezone for post dates, e.g. "Asia/Bangkok" */
  timezone?: string;
  /** Text direction */
  dir?: "ltr" | "rtl" | "auto";
  /** Google Search Console verification meta tag value */
  googleVerification?: string;
}

interface PostsConfig {
  /** Posts per page on paginated listing pages */
  perPage?: number;
  /** Posts shown on the index/home page */
  perIndex?: number;
  /**
   * Scheduled posts within this window (ms) of their pubDatetime
   * are shown as published. Defaults to 15 minutes.
   */
  scheduledPostMargin?: number;
}

interface FeaturesConfig {
  /** Enable light/dark mode toggle. Defaults to true. */
  lightAndDarkMode?: boolean;
  /**
   * Generate dynamic OG images per post and provide `/og.png` when the static
   * `public/{site.ogImage}` file is absent. When false, that file is required
   * for the default layout OG image (build fails if missing).
   */
  dynamicOgImage?: boolean;
  /** Show the /archives page and link it in nav. Defaults to true. */
  showArchives?: boolean;
  /** Show back button on post detail pages. Defaults to true. */
  showBackButton?: boolean;
  /** "Edit page" link shown on post detail pages. */
  editPost?:
    | {
        enabled: true;
        /** Base URL for the edit link, e.g. GitHub edit URL */
        url: string;
      }
    | { enabled: false };
  /**
   * Search provider. "pagefind" ships in the base template.
   * Set to false to disable search entirely.
   */
  search?: "pagefind" | false;
  /**
   * Comment system shown on post detail pages.
   * Set to false to disable comments entirely.
   */
  comments?: CommentsConfig | false;
  /**
   * Site-wide visitor analytics.
   * Set to false to disable analytics entirely.
   */
  analytics?: AnalyticsConfig | false;
}

/**
 * Comments backed by GitHub Discussions via giscus (https://giscus.app).
 *
 * Requires Discussions enabled on `repo` and the giscus GitHub App installed
 * (https://github.com/apps/giscus). Generate `repoId` and `categoryId` on
 * https://giscus.app — both are public identifiers, not secrets.
 */
interface GiscusConfig {
  provider: "giscus";
  /** Repo hosting the Discussions, as "owner/name" */
  repo: `${string}/${string}`;
  /** Repo node ID from giscus.app */
  repoId: string;
  /** Discussion category name, e.g. "Comments" */
  category: string;
  /** Category node ID from giscus.app */
  categoryId: string;
  /** How posts map to discussions. Defaults to "pathname". */
  mapping?: "pathname" | "url" | "title" | "og:title";
  /** Require an exact mapping match instead of a fuzzy search. Defaults to true. */
  strict?: boolean;
  /** Show reactions for the main post. Defaults to true. */
  reactionsEnabled?: boolean;
  /** Place the comment box above or below the thread. Defaults to "bottom". */
  inputPosition?: "top" | "bottom";
  /** giscus theme used in light mode. Defaults to "light". */
  lightTheme?: string;
  /** giscus theme used in dark mode. Defaults to "dark". */
  darkTheme?: string;
  /** giscus UI language. Defaults to site.lang. */
  lang?: string;
}

type CommentsConfig = GiscusConfig;

/**
 * Cookie-free analytics via GoatCounter (https://www.goatcounter.com).
 * No consent banner required. Local/private addresses are filtered out by
 * the vendor script, so `pnpm dev` traffic is never counted.
 */
interface GoatCounterConfig {
  provider: "goatcounter";
  /** Counter endpoint, e.g. "https://mysite.goatcounter.com/count" */
  endpoint: string;
}

type AnalyticsConfig = GoatCounterConfig;

interface SocialLink {
  /**
   * Must match an SVG filename in src/assets/icons/socials/.
   * e.g. "github" → src/assets/icons/socials/github.svg
   */
  name: string;
  url: string;
  /**
   * Accessible label for the icon link (aria-label, title attribute).
   * Auto-generated if omitted: "{site.title} on GitHub", "Send an email to {site.title}", etc.
   * Override when the default wording doesn't fit.
   */
  linkTitle?: string;
}

interface ShareLink {
  /**
   * Must match an SVG filename in src/assets/icons/socials/.
   * e.g. "facebook" → src/assets/icons/socials/facebook.svg
   */
  name: string;
  /** Base share URL. The post URL will be appended as a query param. */
  url: string;
  /**
   * Accessible label for the icon link (aria-label, title attribute).
   * Auto-generated if omitted: "Share this post on Facebook", "Share this post via WhatsApp", etc.
   * Override when the default wording doesn't fit.
   */
  linkTitle?: string;
}

interface AstroPaperConfig {
  site: SiteConfig;
  posts?: PostsConfig;
  features?: FeaturesConfig;
  /** Social profile links shown in header/footer */
  socials?: SocialLink[];
  /** Share links shown on post detail pages */
  shareLinks?: ShareLink[];
}

type ResolvedSiteConfig = Required<
  Pick<
    SiteConfig,
    | "url"
    | "title"
    | "description"
    | "author"
    | "lang"
    | "timezone"
    | "dir"
    | "ogImage"
  >
> &
  Pick<SiteConfig, "profile" | "googleVerification">;

export interface ResolvedAstroPaperConfig {
  site: ResolvedSiteConfig;
  posts: Required<PostsConfig>;
  features: Required<FeaturesConfig>;
  socials: SocialLink[];
  shareLinks: ShareLink[];
}

/**
 * Type helper for astro-paper.config.ts.
 * Provides full IntelliSense without any runtime overhead.
 */
export function defineAstroPaperConfig(
  config: AstroPaperConfig
): AstroPaperConfig {
  return config;
}
