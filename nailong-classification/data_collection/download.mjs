// Re-download the recorded public sources. Node.js >= 18; no login required.
// Run: node data_collection/download.mjs
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
const sources = JSON.parse(await fs.readFile(path.join(here, 'sources.json'), 'utf8'));
const raw = path.join(here, 'raw');
await fs.mkdir(raw, { recursive: true });
const results = [];
let position = 0;
await Promise.all(Array.from({ length: 6 }, async () => {
  while (position < sources.length) {
    const source = sources[position++];
    const target = path.join(raw, source.id);
    try {
      await fs.access(target);
      results.push({ id: source.id, status: 'already_present' });
      continue;
    } catch {}
    try {
      const response = await fetch(source.url, {
        signal: AbortSignal.timeout(15000), headers: { Referer: source.page_url }
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = Buffer.from(await response.arrayBuffer());
      if (data.length > 25_000_000) throw new Error('Image exceeds 25 MB');
      await fs.writeFile(target, data);
      results.push({ id: source.id, status: 'downloaded', bytes: data.length });
    } catch (error) {
      results.push({ id: source.id, status: 'failed', error: String(error) });
    }
  }
}));
await fs.writeFile(path.join(here, 'download_log.json'), JSON.stringify(results, null, 2));
console.log(JSON.stringify(results.reduce((r, x) => ({ ...r, [x.status]: (r[x.status] || 0) + 1 }), {})));
