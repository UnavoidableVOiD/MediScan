// Screenshot pages with real timing (animations settle). Usage: node scripts/shoot.mjs out_dir /path1 /path2 ...
import puppeteer from "puppeteer-core";
import { mkdirSync } from "node:fs";

const [outDir = "shots", ...paths] = process.argv.slice(2);
const base = process.env.BASE_URL ?? "http://127.0.0.1:5173";
const chrome = process.env.CHROME ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
mkdirSync(outDir, { recursive: true });

const browser = await puppeteer.launch({
    executablePath: chrome,
    headless: true,
    args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--hide-scrollbars"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 1200, deviceScaleFactor: 1 });
page.on("pageerror", (e) => console.log("  pageerror:", e.message));
for (const p of paths.length ? paths : ["/"]) {
    await page.goto(base + p, { waitUntil: "networkidle0" });
    await new Promise((r) => setTimeout(r, 2500));
    const name = p === "/" ? "landing" : p.replace(/[/:?=&]+/g, "_").replace(/^_|_$/g, "");
    await page.screenshot({ path: `${outDir}/${name}.png`, fullPage: false });
    console.log("shot", p, "->", `${name}.png`);
}
await browser.close();
