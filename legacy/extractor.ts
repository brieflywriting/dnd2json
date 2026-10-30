import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { z } from "zod";
console.log("starting", process.argv.slice(2));
// ---------- schema ----------
const Ref = z.object({ id: z.string() });
const Named = z.object({ name: z.string(), description: z.string() });
const Feature = z.object({
  id: z.string(),
  name: z.string(),
  level: z.number().int().min(1).max(20),
  description: z.string(),
  options: z.array(Named).optional(),
  refers_to: z.array(z.string()).optional(), // other features mentioned by name
});
const Subclass = z.object({
  id: z.string(),
  parent: Ref,
  name: z.string(),
  version: z.string().optional(),
  description: z.string().min(20),
  features: z.array(Feature).min(1),
  optionalRules: z.array(Named).optional(),
  credits: z.string().optional(),
});

// ---------- helpers ----------
const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
const titleCase = (s: string) =>
  s.toLowerCase()
    .replace(/(^|\s)[a-z]/g, (c) => c.toUpperCase())
    .replace(/\s(Of|The|And|A|To|In|For)\b/g, (w) => w.toLowerCase());
const paragraphs = (s: string) =>
  s.split(/\n\s*\n/).map((p) => p.replace(/\s*\n\s*/g, " ").trim()).filter(Boolean);

const PARENT: Record<string, string> = {
  "roguish archetype": "rogue", "martial archetype": "fighter", "primal path": "barbarian",
  "college of ": "bard", "divine domain": "cleric", "druidic circle": "druid",
  "monastic tradition": "monk", "sacred oath": "paladin", "ranger archetype": "ranger",
  "sorcerous origin": "sorcerer", "otherworldly patron": "warlock", "arcane tradition": "wizard",
};

// ---------- stage 1: OCR cleanup ----------
function clean(t: string): string {
  return t
    .replace(/<div[^>]*>[\s\S]*?<\/div>/g, "")            // image boxes
    .replace(/[“”]/g, '"').replace(/[‘’]/g, "'")
    .replace(/[−–](?=\d)/g, "-")                           // minus signs before numbers
    .replace(/(\w)-\n(\w)/g, "$1$2")                       // hyphenated line breaks
    .replace(/([a-z])\.([A-Z])/g, "$1. $2")                // "Shot.A"
    .replace(/([a-z]),([A-Za-z])/g, "$1, $2")              // "ranged,you"
    .replace(/\b(\d+)(?!st|nd|rd|th)([a-z]{3,})/g, "$1 $2") // "1minute", "0hit"
    .replace(/\b(to|of|than) O\b/g, "$1 0");               // "reduced to O"
}

// ---------- stage 2+3: split and parse ----------
function parse(raw: string) {
  const warnings: string[] = [];
  const chunks = clean(raw).split(/^## /m);
  const top = chunks.shift()!;

  const titleLine = top.split("\n").find((l) => l.startsWith("# ")) ?? "";
  const tm = titleLine.match(/^#\s+(.+?):\s*(.+?)\s*(?:v\s*V?\.?(\d+(?:\.\d+)?))?\s*$/i);
  if (!tm) warnings.push(`could not parse title: "${titleLine}"`);
  const [, prefix = "", rawName = "unknown", version] = tm ?? [];
  const parent = PARENT[prefix.toLowerCase()];
  if (!parent) warnings.push(`unknown parent class prefix: "${prefix}"`);
  const name = titleCase(rawName);

  const description = paragraphs(top.replace(titleLine, "")).join("\n\n");
  if (/^[a-z]/.test(description)) warnings.push("intro starts mid-word (lost drop cap): likely scrambled, fix by hand");

  const features: any[] = [];
  const optionalRules: any[] = [];
  let credits: string | undefined;

  for (const chunk of chunks) {
    const body = chunk.split(/^# /m)[0];                   // cut off the GM Binder footer
    const [h, ...rest] = body.split("\n");
    const heading = titleCase(h.trim());
    const paras = paragraphs(rest.join("\n"));

    if (/^optional rule:/i.test(heading)) {
      optionalRules.push({ name: heading.replace(/^optional rule:\s*/i, ""), description: paras.join("\n\n") });
    } else if (/^created by/i.test(heading)) {
      credits = [heading, ...paras].join(" ");
    } else {
      const lm = paras.join(" ").match(/(\d{1,2})(?:st|nd|rd|th) level/i);
      const options: any[] = [];
      const plain: string[] = [];
      for (const p of paras) {
        const om = p.match(/^([A-Z][a-z']+(?: [A-Z][a-z']+){0,3})\.\s*(\S.*)$/s); // "Bleeding Shot. ..."
        om ? options.push({ name: om[1], description: om[2] }) : plain.push(p);
      }
      features.push({
        id: slug(heading), name: heading, level: lm ? Number(lm[1]) : undefined,
        description: plain.join("\n\n"), ...(options.length && { options }),
      });
    }
  }

  // link features that mention each other by name
  for (const f of features) {
    const text = (f.description + JSON.stringify(f.options ?? "")).toLowerCase();
    const refs = features.filter((o) => o !== f && text.includes(o.name.toLowerCase())).map((o) => o.id);
    if (refs.length) f.refers_to = refs;
  }

  // leftover-artifact scan
  const all = JSON.stringify(features);
  for (const m of all.matchAll(/[A-Za-z']{3,}\.[a-z]{3,}|\b[Ol]\b/g)) warnings.push(`suspicious text: "${m[0]}"`);

  return {
    warnings,
    entry: {
      id: `subclass:${parent ?? "unknown"}-${slug(name)}`,
      parent: { id: `class:${parent ?? "unknown"}` },
      name, version, description, features,
      ...(optionalRules.length && { optionalRules }), credits,
    },
  };
}

// ---------- stage 4: validate + write ----------
const [file] = process.argv.slice(2);
const { entry, warnings } = parse(readFileSync(file, "utf8"));
const result = Subclass.safeParse(entry);
const base = slug(entry.name);
mkdirSync("out", { recursive: true });
mkdirSync("review", { recursive: true });
writeFileSync(`out/${base}.json`, JSON.stringify(entry, null, 2));
if (!result.success || warnings.length) {
  writeFileSync(`review/${base}.json`,
    JSON.stringify({ file, warnings, issues: result.success ? [] : result.error.issues, entry }, null, 2));
  console.log(`⚠ ${entry.name}: ${warnings.length} warnings, ${result.success ? "valid" : "INVALID"}, see review/${base}.json`);
} else console.log(`✓ ${entry.name}`);