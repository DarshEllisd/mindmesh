// Mindmesh front end: one graph, three input modes, links found by embedding similarity.
(() => {
  const STORE_KEY = "mindmesh-v1";
  const LINK_TOP_K = 3; // neighbours checked per new node
  const LINK_MIN_SIM = 0.6; // candidates below this are never sent to the model
  const FALLBACK_SIM = 0.8; // if labelling fails, keep only very close pairs
  const MAX_PAIRS = 30;
  const AUTO_LINK_SIM = 0.8; // this close is the same idea from two sources; no need to ask Gemma
  const ASK_TOP_K = 6;
  const ASK_CACHE_SIM = 0.88; // reworded questions scored 0.92+, unrelated ones about 0.58
  const ASK_CACHE_KEY = "mindmesh-answers";
  const ASK_CACHE_MAX = 30;
  const RESULT_ROWS = 6;
  const ASK_PER_SOURCE = 2; // extra context per source on top of the overall nearest

  const $ = (sel) => document.querySelector(sel);
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // ---------- state ----------
  let mesh = load();
  let mode = "notes";
  const pending = { notes: [], paper: [] }; // image data URLs per mode
  let paperPdf = null;
  const hidden = new Set();

  function load() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORE_KEY));
      if (saved && Array.isArray(saved.nodes)) return saved;
    } catch (_) { /* storage unavailable or corrupt */ }
    return { nodes: [], edges: [] };
  }

  function save() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(mesh));
    } catch (_) {
      setStatus("Could not save to browser storage (it may be full). Export to keep a copy.", "error");
    }
  }

  // ---------- graph ----------
  const visNodes = new vis.DataSet();
  const visEdges = new vis.DataSet();
  const LABEL_FONT = "Archivo Narrow, Archivo, sans-serif";
  const network = new vis.Network($("#graph"), { nodes: visNodes, edges: visEdges }, {
    autoResize: true,
    nodes: { borderWidth: 1.5, font: { size: 14, face: LABEL_FONT } },
    edges: { smooth: { type: "continuous" }, arrows: { to: { enabled: true, scaleFactor: 0.45 } }, font: { size: 11, face: LABEL_FONT, strokeWidth: 4, align: "middle" } },
    physics: { solver: "forceAtlas2Based", forceAtlas2Based: { gravitationalConstant: -130, springLength: 150, avoidOverlap: 0.6 }, stabilization: { iterations: 200 } },
    interaction: { hover: true, tooltipDelay: 200, multiselect: true },
  });

  // Chart graticule drawn in graph coordinates, so it pans and zooms with the map.
  network.on("beforeDrawing", (ctx) => {
    const box = $("#graph").getBoundingClientRect();
    const tl = network.DOMtoCanvas({ x: 0, y: 0 });
    const br = network.DOMtoCanvas({ x: box.width, y: box.height });
    const scale = network.getScale();
    const minorStep = 60, majorEvery = 5;
    const draw = (step, color, width) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = width / scale;
      for (let x = Math.floor(tl.x / step) * step; x <= br.x; x += step) { ctx.moveTo(x, tl.y); ctx.lineTo(x, br.y); }
      for (let y = Math.floor(tl.y / step) * step; y <= br.y; y += step) { ctx.moveTo(tl.x, y); ctx.lineTo(br.x, y); }
      ctx.stroke();
    };
    if (scale > 0.35) draw(minorStep, css("--grid-minor"), 1);
    draw(minorStep * majorEvery, css("--grid-major"), 1);
  });

  const byId = (id) => mesh.nodes.find((n) => n.id === id);
  const wrap = (s, w = 24) => s.replace(new RegExp(`(.{1,${w}})(\\s|$)`, "g"), "$1\n").trim();
  const SOURCE_PHRASE = { notes: "your notes", repo: "the repo", paper: "the paper" };
  const MODE_SHAPE = { notes: "dot", repo: "square", paper: "triangle" };
  const KIND_SHAPE = { issue: "diamond", repository: "hexagon" };
  const capital = (s) => s.charAt(0).toUpperCase() + s.slice(1);

  function nodeView(n) {
    const color = css(`--src-${n.mode}`);
    const root = n.kind === "repository" || (n.mode === "paper" && n.extra && n.extra.root);
    const order = n.extra && n.extra.order;
    return {
      id: n.id,
      label: wrap((order ? `${order}. ` : "") + n.title),
      title: `${capital(n.kind)} from ${SOURCE_PHRASE[n.mode] || n.mode}`,
      shape: root && n.mode === "paper" ? "star" : KIND_SHAPE[n.kind] || MODE_SHAPE[n.mode] || "dot",
      size: root ? 20 : n.kind === "issue" ? 8 : 11,
      color: {
        background: color,
        border: css("--sheet"),
        highlight: { background: color, border: css("--ink") },
        hover: { background: color, border: css("--ink") },
      },
      font: { color: css("--ink"), strokeWidth: 4, strokeColor: css("--sheet"), bold: { face: LABEL_FONT } },
      borderWidthSelected: 3,
      hidden: hidden.has(n.mode),
    };
  }

  function edgeView(e) {
    const a = byId(e.source), b = byId(e.target);
    const cross = a && b && a.mode !== b.mode;
    const color = cross ? css("--route") : css("--rule-strong");
    return {
      id: e.id,
      from: e.source,
      to: e.target,
      // Only cross-source links are labelled on the map; every label is still listed in the detail panel.
      label: cross ? e.label || "" : undefined,
      title: e.label || undefined,
      dashes: !cross && e.kind === "semantic" ? [6, 5] : false,
      width: cross ? 2.6 : 1.1,
      color: { color, highlight: cross ? color : css("--ink"), hover: cross ? color : css("--ink") },
      font: { color: cross ? css("--route") : css("--ink-soft"), strokeColor: css("--sheet") },
    };
  }

  function render() {
    visNodes.clear();
    visEdges.clear();
    visNodes.add(mesh.nodes.map(nodeView));
    visEdges.add(mesh.edges.map(edgeView));
    refreshMeta();
  }

  function refreshMeta() {
    const modeOf = Object.fromEntries(mesh.nodes.map((n) => [n.id, n.mode]));
    const across = mesh.edges.filter((e) => modeOf[e.source] && modeOf[e.source] !== modeOf[e.target]).length;
    const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
    const format = (v) => (v
      ? `${plural(v.ideas, "idea")}, ${plural(v.links, "link")}, ${v.across} across sources`
      : "Your map is saved in this browser");
    Motion.counts($("#counts"), mesh.nodes.length ? { ideas: mesh.nodes.length, links: mesh.edges.length, across } : null, format);
    $("#empty").hidden = mesh.nodes.length > 0;
  }

  function addGraph(graph, kind) {
    const known = new Set(mesh.nodes.map((n) => n.id));
    const nodes = graph.nodes.filter((n) => !known.has(n.id));
    const ids = new Set([...known, ...nodes.map((n) => n.id)]);
    const edges = (graph.edges || [])
      .filter((e) => ids.has(e.source) && ids.has(e.target))
      .filter((e) => !mesh.edges.some((x) => x.source === e.source && x.target === e.target))
      .map((e, i) => ({ ...e, id: `e_${Date.now().toString(36)}_${i}_${Math.random().toString(36).slice(2, 6)}`, kind }));
    mesh.nodes.push(...nodes);
    mesh.edges.push(...edges);
    const nodeViews = nodes.map(nodeView);
    const edgeViews = edges.map(edgeView);
    visNodes.add(nodeViews);
    visEdges.add(edgeViews);
    Motion.plot(visNodes, nodeViews);
    Motion.routes(visEdges, edgeViews.filter((v) => v.label !== undefined));
    refreshMeta();
    save();
    return { nodes, edges };
  }

  // ---------- semantic linking ----------
  const dot = (a, b) => {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  };

  // Compare only new nodes against existing ones, then ask the model to label just those top-k
  // pairs. The whole board is never sent to the model.
  async function linkNew(newNodes) {
    const newIds = new Set(newNodes.map((n) => n.id));
    const old = mesh.nodes.filter((n) => !newIds.has(n.id) && n.vec);
    if (!old.length) return 0;
    const pairs = [];
    for (const n of newNodes) {
      if (!n.vec) continue;
      old.map((o) => ({ o, sim: dot(n.vec, o.vec) }))
        .filter((x) => x.sim >= LINK_MIN_SIM)
        .sort((x, y) => y.sim - x.sim)
        .slice(0, LINK_TOP_K)
        .forEach((x) => pairs.push({ a: n, b: x.o, sim: x.sim }));
    }
    pairs.sort((x, y) => y.sim - x.sim);
    if (!pairs.length) return 0;

    // Near-identical pairs are linked straight from their embeddings; only the uncertain middle
    // band goes to Gemma, and the call is skipped when there is nothing uncertain left.
    const sure = pairs.filter((p) => p.sim >= AUTO_LINK_SIM);
    const unsure = pairs.filter((p) => p.sim < AUTO_LINK_SIM).slice(0, MAX_PAIRS);
    const edges = sure.map((p) => ({ source: p.a.id, target: p.b.id, label: "same idea" }));
    if (unsure.length) {
      const brief = (n) => ({ id: n.id, title: n.title, body: n.body, mode: n.mode });
      try {
        const res = await api("/api/relate", { pairs: unsure.map((p) => ({ a: brief(p.a), b: brief(p.b) })) });
        edges.push(...res.edges);
      } catch (_) {
        unsure.filter((p) => p.sim >= FALLBACK_SIM).forEach((p) => edges.push({ source: p.a.id, target: p.b.id, label: "similar" }));
      }
    }
    return addGraph({ nodes: [], edges }, "semantic").edges.length;
  }

  // ---------- API ----------
  async function api(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    let data = null;
    try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
    if (!res.ok) throw new Error((data && data.detail) || `Request failed (${res.status})`);
    return data;
  }

  function setStatus(msg, kind) {
    const el = $("#status");
    el.className = kind || "";
    if (!msg) { el.textContent = ""; return; }
    el.innerHTML = kind === "busy" ? `<span class="spinner"></span><span></span>` : "<span></span>";
    el.lastChild.textContent = msg;
  }

  async function ingest(runMode) {
    const btn = document.querySelector(`[data-run="${runMode}"]`);
    let payload, endpoint = `/api/${runMode}`;
    if (runMode === "notes") {
      payload = { text: $("#notes-text").value, images: pending.notes };
      if (!payload.text.trim() && !payload.images.length) return setStatus("Add some text or an image first.", "error");
    } else if (runMode === "repo") {
      payload = { url: $("#repo-url").value.trim() };
      if (!payload.url) return setStatus("Enter a GitHub repository URL.", "error");
    } else {
      payload = { pdf: paperPdf, text: $("#paper-text").value, images: pending.paper };
      if (!payload.pdf && !payload.text.trim() && !payload.images.length) return setStatus("Choose a PDF, paste text, or add figures.", "error");
    }

    btn.disabled = true;
    Motion.busy(true);
    const started = performance.now();
    setStatus({ notes: "Gemma 4 is reading your notes…", repo: "Fetching the repo and mapping it with Gemma 4…", paper: "Reading the paper with Gemma 4…" }[runMode], "busy");
    try {
      const graph = await api(endpoint, payload);
      if (runMode === "paper" && graph.nodes[0]) graph.nodes[0].extra = { ...graph.nodes[0].extra, root: true };
      const added = addGraph(graph, "ai");
      setStatus(`Mapped ${added.nodes.length} ideas. Checking how they connect to the rest of your map…`, "busy");
      const links = await linkNew(added.nodes);
      const secs = ((performance.now() - started) / 1000).toFixed(1);
      setStatus(`Mapped ${added.nodes.length} ideas and ${added.edges.length + links} links in ${secs}s. ${links ? `${links} of them connect to what was already on the map.` : "None connect to earlier ideas yet."}`);
      network.fit({ nodes: added.nodes.map((n) => n.id), animation: { duration: 600 } });
      clearInputs(runMode);
    } catch (err) {
      setStatus(err.message, "error");
    } finally {
      btn.disabled = false;
      Motion.busy(false);
    }
  }

  function clearInputs(runMode) {
    if (runMode === "notes") {
      $("#notes-text").value = "";
      pending.notes = [];
      renderThumbs("notes");
    } else if (runMode === "paper") {
      $("#paper-text").value = "";
      pending.paper = [];
      paperPdf = null;
      $("#paper-file").textContent = "";
      renderThumbs("paper");
    }
  }

  // ---------- Ask: search with embeddings first, Gemma explains on request ----------
  let lastSearch = null; // { q, qVec, context }

  function loadAnswers() {
    try { return JSON.parse(localStorage.getItem(ASK_CACHE_KEY)) || []; } catch (_) { return []; }
  }

  function saveAnswer(entry) {
    try {
      const list = [entry, ...loadAnswers()].slice(0, ASK_CACHE_MAX);
      localStorage.setItem(ASK_CACHE_KEY, JSON.stringify(list));
    } catch (_) { /* caching is best effort */ }
  }

  // A cached answer is reused when the question means the same thing and its citations still exist.
  function cachedAnswer(qVec) {
    let best = null;
    for (const entry of loadAnswers()) {
      const sim = dot(qVec, entry.qVec);
      if (sim >= ASK_CACHE_SIM && entry.cited.every((id) => byId(id)) && (!best || sim > best.sim)) best = { ...entry, sim };
    }
    return best;
  }

  async function ask() {
    const q = $("#ask-input").value.trim();
    if (!q) return;
    const withVec = mesh.nodes.filter((n) => n.vec);
    if (!withVec.length) return setStatus("Add something to the map before asking.", "error");
    $("#ask-btn").disabled = true;
    Motion.busy(true);
    setStatus("Searching your map…", "busy");
    $("#answer").hidden = true;
    try {
      const { vectors } = await api("/api/embed", { texts: [q] });
      const qVec = vectors[0];
      // Nearest ideas overall, plus the best few from each source, so a question about code
      // still reaches the repo even when the paper and notes score higher.
      const ranked = withVec.map((n) => ({ n, sim: dot(qVec, n.vec) })).sort((a, b) => b.sim - a.sim);
      const context = ranked.slice(0, ASK_TOP_K).map((x) => x.n);
      ["notes", "repo", "paper"].forEach((m) => {
        ranked.filter((x) => x.n.mode === m && !context.includes(x.n)).slice(0, ASK_PER_SOURCE).forEach((x) => context.push(x.n));
      });
      lastSearch = { q, qVec, context };
      showResults(ranked.slice(0, RESULT_ROWS));
      setStatus("");

      const cached = cachedAnswer(qVec);
      if (cached) {
        showAnswer(cached.answer, cached.cited, cached.q === q ? "Answered earlier" : `Answered earlier for “${cached.q}”`);
        setExplain("Ask Gemma again");
        focusOn(cached.cited.length ? cached.cited : context.map((n) => n.id));
      } else {
        setExplain("Explain with Gemma");
        focusOn(ranked.slice(0, RESULT_ROWS).map((x) => x.n.id));
      }
    } catch (err) {
      setStatus(err.message, "error");
    } finally {
      $("#ask-btn").disabled = false;
      Motion.busy(false);
    }
  }

  async function explain() {
    if (!lastSearch) return;
    const { q, qVec, context } = lastSearch;
    const btn = $("#explain-btn");
    btn.disabled = true;
    Motion.busy(true);
    setStatus("Gemma 4 is writing an answer from these ideas…", "busy");
    try {
      const res = await api("/api/ask", { question: q, context: context.map(askContext) });
      showAnswer(res.answer, res.cited);
      saveAnswer({ q, qVec, answer: res.answer, cited: res.cited });
      setStatus("");
      btn.hidden = true;
      if (res.cited.length) focusOn(res.cited);
    } catch (err) {
      setStatus(err.message, "error");
    } finally {
      btn.disabled = false;
      Motion.busy(false);
    }
  }

  function setExplain(label) {
    const btn = $("#explain-btn");
    btn.textContent = label;
    btn.hidden = false;
  }

  function focusOn(ids) {
    network.selectNodes(ids);
    network.fit({ nodes: ids, animation: { duration: 600 } });
  }

  function showResults(rows) {
    const list = $("#results");
    list.textContent = "";
    rows.forEach(({ n }) => {
      const li = document.createElement("li");
      const row = document.createElement("button");
      row.className = "result";
      const glyph = document.createElement("span");
      glyph.className = `glyph glyph-${n.mode}`;
      glyph.setAttribute("aria-hidden", "true");
      const text = document.createElement("span");
      const title = document.createElement("span");
      title.className = "result-title";
      title.textContent = n.title;
      const snippet = document.createElement("span");
      snippet.className = "result-snippet";
      snippet.textContent = n.body.length > 90 ? n.body.slice(0, 88).trimEnd() + "…" : n.body;
      text.append(title, snippet);
      row.append(glyph, text);
      row.onclick = () => focusNode(n.id);
      li.appendChild(row);
      list.appendChild(li);
    });
    list.hidden = !rows.length;
    Motion.results(list);
  }

  // Repo nodes carry their file path so answers can point at real code.
  function askContext(n) {
    const path = n.extra && n.extra.path ? ` (file: ${n.extra.path})` : "";
    return { id: n.id, title: n.title, body: n.body + path, mode: n.mode };
  }

  function showAnswer(text, cited, note) {
    const box = $("#answer");
    box.hidden = false;
    box.textContent = "";
    if (note) {
      const p = document.createElement("span");
      p.className = "answer-note";
      p.textContent = note;
      box.appendChild(p);
    }
    // Turn [node-id] citations into clickable chips.
    text.split(/(\[[\w-]+\])/g).forEach((part) => {
      const m = part.match(/^\[([\w-]+)\]$/);
      const node = m && byId(m[1]);
      if (node) {
        const chip = document.createElement("span");
        chip.className = `cite src-${node.mode}`;
        chip.textContent = node.title;
        chip.title = node.title;
        chip.onclick = () => focusNode(node.id);
        box.appendChild(chip);
      } else {
        box.appendChild(document.createTextNode(part));
      }
    });
    Motion.answer(box);
  }

  // ---------- detail panel ----------
  function focusNode(id) {
    network.selectNodes([id]);
    network.focus(id, { scale: 1.1, animation: { duration: 500 } });
    showDetail(id);
  }

  function showDetail(id) {
    const n = byId(id);
    if (!n) return;
    const el = $("#detail-body");
    const wasHidden = $("#detail").hidden;
    el.textContent = "";
    const add = (tag, cls, text) => {
      const x = document.createElement(tag);
      if (cls) x.className = cls;
      if (text != null) x.textContent = text;
      el.appendChild(x);
      return x;
    };
    const ex = n.extra || {};
    add("p", "provenance", `${capital(n.kind)} from ${SOURCE_PHRASE[n.mode] || n.mode}`);
    add("h3", "", n.title);
    if (ex.order) add("p").appendChild(Object.assign(document.createElement("span"), { className: "step", textContent: `Step ${Number(ex.order)} of the reading order` }));
    if (n.body) add("p", "", n.body);
    if (ex.path) add("p", "meta", `File: ${ex.path}`);
    if (ex.license) add("p", "meta", `License ${ex.license}${ex.stars != null ? `, ${ex.stars.toLocaleString()} stars` : ""}`);
    if (n.tags && n.tags.length) add("p", "meta", n.tags.map((t) => `#${t}`).join("  "));
    if (ex.url) {
      const p = add("p");
      const a = document.createElement("a");
      a.href = ex.url;
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = n.kind === "issue" ? "Open issue on GitHub" : "Open on GitHub";
      p.appendChild(a);
    }

    const conns = mesh.edges.filter((e) => e.source === id || e.target === id);
    if (conns.length) {
      add("h4", "conn-head", conns.length === 1 ? "1 connection" : `${conns.length} connections`);
      conns.forEach((e) => {
        const other = byId(e.source === id ? e.target : e.source);
        if (!other) return;
        const row = add("button", "conn");
        const rel = document.createElement("span");
        rel.className = "rel" + (other.mode !== n.mode ? " route" : "");
        rel.textContent = (e.source === id ? "→ " : "← ") + (e.label || "linked");
        const t = document.createElement("span");
        t.className = "t";
        t.textContent = other.mode !== n.mode ? `${other.title}, in ${SOURCE_PHRASE[other.mode]}` : other.title;
        row.append(rel, t);
        row.onclick = () => focusNode(other.id);
      });
    }
    const del = add("button", "text-btn danger delete", "Remove from map");
    del.onclick = () => removeNodes([id]);
    $("#detail").className = `src-${n.mode}`;
    $("#detail").hidden = false;
    wasHidden ? Motion.detailIn($("#detail")) : Motion.detailSwap(el);
  }

  function closeDetail() {
    const panel = $("#detail");
    if (panel.hidden) return;
    Motion.detailOut(panel, () => { panel.hidden = true; });
  }

  network.on("click", (params) => {
    if (params.nodes.length === 1) showDetail(params.nodes[0]);
    else if (!params.nodes.length) closeDetail();
    updateSelection(params.nodes);
  });
  $("#detail-close").onclick = closeDetail;

  // ---------- selecting and removing several ideas ----------
  // Ctrl/Cmd-click toggles ideas (vis-network multiselect); Shift-drag draws a selection box.
  // Removal is undoable for a few seconds instead of asking for confirmation.
  const UNDO_MS = 8000;
  let undoState = null;
  let undoTimer = null;

  function showBar(mode, text) {
    const bar = $("#selection-bar");
    $("#selection-count").textContent = text;
    $("#selection-remove").hidden = mode !== "select";
    $("#selection-clear").hidden = mode !== "select";
    $("#selection-undo").hidden = mode !== "undo";
    const wasHidden = bar.hidden;
    bar.hidden = false;
    if (wasHidden) Motion.panel(bar);
  }

  function updateSelection(ids) {
    if (ids.length > 1) {
      closeDetail();
      showBar("select", `${ids.length} ideas selected`);
    } else if (!undoState) {
      $("#selection-bar").hidden = true;
    }
  }

  function clearSelection() {
    network.unselectAll();
    updateSelection([]);
  }

  function removeNodes(ids) {
    if (!ids.length) return;
    const gone = new Set(ids);
    const nodes = mesh.nodes.filter((n) => gone.has(n.id));
    const edges = mesh.edges.filter((e) => gone.has(e.source) || gone.has(e.target));
    mesh.nodes = mesh.nodes.filter((n) => !gone.has(n.id));
    mesh.edges = mesh.edges.filter((e) => !gone.has(e.source) && !gone.has(e.target));
    save();
    render();
    closeDetail();
    undoState = { nodes, edges };
    showBar("undo", nodes.length === 1 ? `Removed “${nodes[0].title}”` : `Removed ${nodes.length} ideas`);
    clearTimeout(undoTimer);
    undoTimer = setTimeout(() => {
      undoState = null;
      updateSelection(network.getSelectedNodes());
    }, UNDO_MS);
  }

  function undoRemove() {
    if (!undoState) return;
    clearTimeout(undoTimer);
    mesh.nodes.push(...undoState.nodes);
    const ids = new Set(mesh.nodes.map((n) => n.id));
    mesh.edges.push(...undoState.edges.filter((e) => ids.has(e.source) && ids.has(e.target)));
    const restored = undoState.nodes.map((n) => n.id);
    undoState = null;
    save();
    render();
    $("#selection-bar").hidden = true;
    network.selectNodes(restored);
    updateSelection(restored);
  }

  $("#selection-remove").onclick = () => removeNodes(network.getSelectedNodes());
  $("#selection-clear").onclick = clearSelection;
  $("#selection-undo").onclick = undoRemove;

  const isTyping = (e) => e.target.closest && e.target.closest("input, textarea, [contenteditable]");
  document.addEventListener("keydown", (e) => {
    if (isTyping(e)) return;
    if (e.key === "Delete" || e.key === "Backspace") {
      const ids = network.getSelectedNodes();
      if (ids.length) {
        e.preventDefault();
        removeNodes(ids);
      }
    } else if (e.key === "Escape") {
      clearSelection();
      closeDetail();
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && undoState) {
      e.preventDefault();
      undoRemove();
    }
  });

  // Shift-drag selection box in the chart's route colour. vis-network has its own (hard-coded blue)
  // box on Shift-drag, so Shift presses are stopped here before they reach it.
  const graphEl = $("#graph");
  const lasso = $("#lasso");
  let lassoStart = null;
  const local = (e) => {
    const box = graphEl.getBoundingClientRect();
    return { x: e.clientX - box.left, y: e.clientY - box.top };
  };
  const rectOf = (a, b) => ({ x1: Math.min(a.x, b.x), y1: Math.min(a.y, b.y), x2: Math.max(a.x, b.x), y2: Math.max(a.y, b.y) });

  graphEl.addEventListener("pointerdown", (e) => {
    if (!e.shiftKey || e.button !== 0) return;
    e.preventDefault();
    e.stopPropagation();
    lassoStart = local(e);
    Object.assign(lasso.style, { left: `${lassoStart.x}px`, top: `${lassoStart.y}px`, width: "0px", height: "0px" });
    lasso.hidden = false;
  }, true);

  window.addEventListener("pointermove", (e) => {
    if (!lassoStart) return;
    const r = rectOf(lassoStart, local(e));
    Object.assign(lasso.style, { left: `${r.x1}px`, top: `${r.y1}px`, width: `${r.x2 - r.x1}px`, height: `${r.y2 - r.y1}px` });
  });

  window.addEventListener("pointerup", (e) => {
    if (!lassoStart) return;
    const r = rectOf(lassoStart, local(e));
    lassoStart = null;
    lasso.hidden = true;
    if (r.x2 - r.x1 < 4 && r.y2 - r.y1 < 4) return;
    const pos = network.getPositions();
    const inside = mesh.nodes
      .filter((n) => !hidden.has(n.mode) && pos[n.id])
      .filter((n) => {
        const p = network.canvasToDOM(pos[n.id]);
        return p.x >= r.x1 && p.x <= r.x2 && p.y >= r.y1 && p.y <= r.y2;
      })
      .map((n) => n.id);
    const ids = [...new Set([...network.getSelectedNodes(), ...inside])];
    network.selectNodes(ids);
    if (ids.length === 1) showDetail(ids[0]);
    updateSelection(ids);
  });

  // ---------- inputs ----------
  function downscale(file, max = 1280) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(1, max / Math.max(img.width, img.height));
        const c = document.createElement("canvas");
        c.width = Math.round(img.width * scale);
        c.height = Math.round(img.height * scale);
        c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
        URL.revokeObjectURL(img.src);
        resolve(c.toDataURL("image/jpeg", 0.82));
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }

  async function addImages(target, files) {
    for (const f of files) {
      if (!f.type.startsWith("image/")) continue;
      if (pending[target].length >= 4) { setStatus("Up to 4 images per request.", "error"); break; }
      pending[target].push(await downscale(f));
    }
    renderThumbs(target);
  }

  function renderThumbs(target) {
    const box = $(`#${target}-thumbs`);
    box.textContent = "";
    pending[target].forEach((src, i) => {
      const t = document.createElement("div");
      t.className = "thumb";
      const img = document.createElement("img");
      img.src = src;
      img.alt = `image ${i + 1}`;
      const x = document.createElement("button");
      x.textContent = "×";
      x.setAttribute("aria-label", "Remove image");
      x.onclick = () => { pending[target].splice(i, 1); renderThumbs(target); };
      t.append(img, x);
      box.appendChild(t);
    });
  }

  $("#notes-images").onchange = (e) => addImages("notes", e.target.files);
  $("#paper-images").onchange = (e) => addImages("paper", e.target.files);
  $("#notes-text").addEventListener("paste", (e) => {
    const files = [...(e.clipboardData.files || [])];
    if (files.length) { e.preventDefault(); addImages("notes", files); }
  });
  $("#paper-pdf").onchange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    if (f.size > 3.2 * 1024 * 1024) {
      setStatus("That PDF is over about 3MB. Try a smaller file or paste the text.", "error");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => { paperPdf = reader.result; $("#paper-file").textContent = `📄 ${f.name} (${(f.size / 1024).toFixed(0)} KB)`; };
    reader.readAsDataURL(f);
  };

  const SAMPLES = {
    notes: `ML lecture 7 - transformers
- RNNs process tokens one by one -> slow, hard to parallelise, forget long context
- self-attention: every token looks at every other token, weights = softmax(QK^T / sqrt(d))
- multi-head attention = several attention ops in parallel, each learns a different pattern
- positional encoding needed because attention has no sense of order
- GPT = decoder-only transformer trained to predict the next token
- TODO: read the original "Attention Is All You Need" paper and try training a tiny GPT on Shakespeare`,
    repo: "https://github.com/karpathy/nanoGPT",
  };
  document.querySelectorAll("[data-sample]").forEach((b) => {
    b.onclick = () => {
      if (b.dataset.sample === "notes") $("#notes-text").value = SAMPLES.notes;
      else $("#repo-url").value = SAMPLES.repo;
    };
  });

  document.querySelectorAll(".source").forEach((t) => {
    t.onclick = () => {
      mode = t.dataset.mode;
      document.querySelectorAll(".source").forEach((x) => {
        x.classList.toggle("active", x === t);
        x.setAttribute("aria-selected", String(x === t));
      });
      document.querySelectorAll("[data-panel]").forEach((p) => { p.hidden = p.dataset.panel !== mode; });
      Motion.tab(t);
      Motion.panel(document.querySelector(`[data-panel="${mode}"]`));
    };
  });
  document.querySelectorAll("[data-run]").forEach((b) => { b.onclick = () => ingest(b.dataset.run); });
  $("#repo-url").addEventListener("keydown", (e) => { if (e.key === "Enter") ingest("repo"); });
  $("#ask-btn").onclick = ask;
  $("#explain-btn").onclick = explain;
  $("#ask-input").addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });

  document.querySelectorAll("[data-filter]").forEach((cb) => {
    cb.onchange = () => {
      cb.checked ? hidden.delete(cb.dataset.filter) : hidden.add(cb.dataset.filter);
      visNodes.update(mesh.nodes.map((n) => ({ id: n.id, hidden: hidden.has(n.mode) })));
    };
  });

  $("#export-btn").onclick = () => {
    const blob = new Blob([JSON.stringify(mesh, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "mindmesh.json";
    a.click();
    URL.revokeObjectURL(a.href);
  };
  $("#clear-btn").onclick = () => {
    if (!mesh.nodes.length || !confirm("Clear the whole mesh? This cannot be undone.")) return;
    mesh = { nodes: [], edges: [] };
    save();
    render();
    $("#detail").hidden = true;
    $("#answer").hidden = true;
    $("#results").hidden = true;
    $("#explain-btn").hidden = true;
    $("#selection-bar").hidden = true;
    undoState = null;
    lastSearch = null;
    try { localStorage.removeItem(ASK_CACHE_KEY); } catch (_) { /* nothing cached */ }
  };

  // Re-colour when the system theme flips.
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    render();
    Motion.tab(document.querySelector(".source.active"), true);
  });
  window.addEventListener("resize", () => Motion.tab(document.querySelector(".source.active"), true));

  $("#empty-sample").onclick = () => {
    $("#notes-text").value = SAMPLES.notes;
    document.querySelector('.source[data-mode="notes"]').click();
    ingest("notes");
  };

  render();
  Motion.intro();
  Motion.tab(document.querySelector(".source.active"), true);
  network.once("stabilizationIterationsDone", () => network.fit({ animation: false }));
  // Canvas labels need the web font loaded before they are drawn.
  if (document.fonts) document.fonts.ready.then(() => { network.redraw(); network.fit({ animation: false }); });
  // Handle for debugging from the browser console.
  window.mindmesh = { network, mesh: () => mesh };
})();
