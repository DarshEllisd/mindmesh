// Motion layer (GSAP). One orchestrated intro; everything else answers a user action.
// app.js calls window.Motion hooks; when GSAP is missing or reduced motion is on they do the minimum.
(() => {
  const g = window.gsap;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const live = Boolean(g) && !reduce;

  const noop = () => {};
  const instant = (el, done) => { if (done) done(); };

  if (!live) {
    window.Motion = {
      intro: noop, tab: noop, panel: noop, detailIn: noop, detailSwap: noop, detailOut: instant,
      answer: noop, results: noop, plot: noop, routes: noop, counts: (el, values, format) => { el.textContent = format(values); },
      busy: (on) => { const bar = document.getElementById("progress"); if (bar) bar.hidden = !on; },
    };
    return;
  }

  g.defaults({ ease: "power3.out", duration: 0.5 });

  // ---------- page load: the wordmark is plotted from narrow to wide ----------
  function intro() {
    const visibleEmpty = !document.getElementById("empty").hidden;
    const tl = g.timeline({ defaults: { ease: "expo.out" } });
    tl.fromTo(".wordmark", { "--wdth": 62, opacity: 0, letterSpacing: "0.08em" },
      { "--wdth": 125, opacity: 1, letterSpacing: "-0.01em", duration: 1.3 })
      .from(".tagline", { opacity: 0, y: 8, duration: 0.6 }, "-=0.9")
      .from(["#dock .sources", "#dock .panel:not([hidden])", "#dock .ask", "#dock .dock-foot"],
        { opacity: 0, y: 12, stagger: 0.07, duration: 0.6, clearProps: "transform,opacity" }, "-=0.7")
      .from("#graph", { opacity: 0, duration: 0.9, ease: "power2.out" }, "-=0.6")
      .from(".legend", { opacity: 0, y: 18, duration: 0.6, clearProps: "transform,opacity" }, "-=0.6");
    if (visibleEmpty) tl.from("#empty", { opacity: 0, y: 16, scale: 0.98, duration: 0.7, clearProps: "transform,opacity" }, "-=0.4");
  }

  // ---------- source switcher: one indicator slides to the active source ----------
  let indicator;
  function tab(button, first) {
    const bar = button.parentElement;
    if (!indicator) {
      indicator = document.createElement("span");
      indicator.className = "sources-indicator";
      bar.appendChild(indicator);
      bar.classList.add("has-indicator");
    }
    const color = getComputedStyle(document.documentElement).getPropertyValue(`--src-${button.dataset.mode}`).trim();
    const vars = { x: button.offsetLeft, width: button.offsetWidth, backgroundColor: color };
    first ? g.set(indicator, vars) : g.to(indicator, { ...vars, duration: 0.45, ease: "power4.out" });
  }

  function panel(el) {
    g.fromTo(el, { opacity: 0, y: 8 }, { opacity: 1, y: 0, duration: 0.35, clearProps: "transform,opacity" });
  }

  // ---------- detail card ----------
  function detailIn(el) {
    g.killTweensOf(el);
    g.fromTo(el, { opacity: 0, x: 28 }, { opacity: 1, x: 0, duration: 0.45, clearProps: "transform,opacity" });
    detailSwap(el.querySelector("#detail-body"));
  }

  function detailSwap(body) {
    g.fromTo(body.children, { opacity: 0, y: 6 }, { opacity: 1, y: 0, stagger: 0.03, duration: 0.35, clearProps: "transform,opacity" });
  }

  function detailOut(el, done) {
    g.to(el, { opacity: 0, x: 28, duration: 0.25, ease: "power2.in", onComplete: () => { done(); g.set(el, { clearProps: "transform,opacity" }); } });
  }

  // ---------- answer: unfolds, then citations land one by one ----------
  function answer(box) {
    g.fromTo(box, { height: 0, opacity: 0 }, { height: "auto", opacity: 1, duration: 0.5, clearProps: "height,opacity" });
    g.from(box.querySelectorAll(".cite"), { opacity: 0, y: 6, scale: 0.85, stagger: 0.08, delay: 0.25, duration: 0.4, ease: "back.out(2)", clearProps: "all" });
  }

  function results(list) {
    g.fromTo(list.children, { opacity: 0, x: -8 }, { opacity: 1, x: 0, stagger: 0.04, duration: 0.3, clearProps: "transform,opacity" });
  }

  // ---------- map: new ideas are plotted point by point ----------
  function plot(dataset, views) {
    if (!views.length) return;
    const scale = views.map(() => ({ s: 0.05 }));
    dataset.update(views.map((v) => ({ id: v.id, size: v.size * 0.05 })));
    g.to(scale, {
      s: 1,
      duration: 0.7,
      ease: "back.out(2.2)",
      stagger: 0.05,
      onUpdate: () => dataset.update(views.map((v, i) => ({ id: v.id, size: v.size * scale[i].s }))),
    });
  }

  // Cross-source routes pulse once when Gemma confirms them.
  function routes(dataset, views) {
    if (!views.length) return;
    const w = { k: 0 };
    g.to(w, {
      k: 1,
      duration: 0.55,
      ease: "sine.inOut",
      yoyo: true,
      repeat: 3,
      onUpdate: () => dataset.update(views.map((v) => ({ id: v.id, width: v.width + 5 * w.k }))),
      onComplete: () => dataset.update(views.map((v) => ({ id: v.id, width: v.width }))),
    });
  }

  // ---------- counters count up to their new values ----------
  let shown = null;
  let counter;
  function counts(el, values, format) {
    if (counter) counter.kill();
    if (!values || !shown) {
      el.textContent = format(values);
      shown = values;
      return;
    }
    const state = { ...shown };
    shown = values;
    counter = g.to(state, {
      ...values,
      duration: 0.9,
      ease: "power2.out",
      onUpdate: () => { el.textContent = format(Object.fromEntries(Object.entries(state).map(([k, v]) => [k, Math.round(v)]))); },
    });
  }

  // ---------- long waits: a plotter head sweeps under the status line ----------
  let sweep;
  function busy(on) {
    const bar = document.getElementById("progress");
    if (!bar) return;
    if (on) {
      bar.hidden = false;
      sweep = g.fromTo(bar.firstElementChild, { xPercent: -110 }, { xPercent: 330, duration: 1.4, ease: "power1.inOut", repeat: -1 });
    } else {
      if (sweep) sweep.kill();
      bar.hidden = true;
    }
  }

  window.Motion = { intro, tab, panel, detailIn, detailSwap, detailOut, answer, results, plot, routes, counts, busy };
})();
