const seen = new Set<string>();

/**
 * Log a build-time warning at most once per message.
 *
 * Astro component frontmatter re-runs for every page, so a bare `console.warn`
 * in a layout-level component repeats once per route. Module scope persists
 * across those renders, so the Set dedupes them.
 */
export function warnOnce(message: string): void {
  if (seen.has(message)) return;
  seen.add(message);
  // Deliberate: this is build-time diagnostic output, not stray debug logging.
  // eslint-disable-next-line no-console
  console.warn(message);
}
