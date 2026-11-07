# What the interface must do: the judge of the GUI exploration

This file holds the user's own rules, gathered from his feedback. An exploring agent checks the interface against it,
not against what the code intends. Anything that breaks a rule, or that a user would find odd, is an anomaly to note.
When in doubt, note it: the user judges.

## The model he has in mind

- **The workflow is the whole graph on screen.** Chains, flows and branches are parts of it. A source block (Session,
  Image, Fusion) is a block like the others: "Delete block", never "Delete workflow".
- **The graph is built block by block from a palette.** A block is dropped by drag and drop or a double-click. It
  places and links itself; the user links by hand only when there are several options. While dragging, only the
  valid places light up.
- **Templates appear once, when the project is made.** After that, no action opens the template window: not
  dropping or double-clicking a block (Session, Image, Fusion...), not Duplicate, not the bottom zone.
- **Types decide what fits.** Frame blocks (Calibration, Registration, Selection, Stacking) take frames; image blocks
  take an image. A block is greyed in the palette when it cannot go in the chosen place. A small message says why,
  in plain words and in the interface's language.
- **Far dependencies.** A block that needs a block far upstream (Color needs Astrometry) is refused with the reason.
- **Every block that can go has its cross**, the Del key and "Delete block" in its menu. Deleting a block that the
  others need is refused with the reason. A placed block dragged back onto the palette is removed.
- **Switching off** (D, or the menu) keeps a block's settings and passes the data through. It is offered only where
  the block's output kind equals its input kind; elsewhere the menu says why.
- **Some blocks can be placed twice** (Gradient, Denoise, Color). The copies are numbered left to right ("Gradient 1",
  "Gradient 2"), with the same name everywhere: graph, panel, run window, messages.
- **Any block giving an image sends its image to Post**: from its menu, or by pulling its port onto Post.
- **Parallel flows and branches.** A Session or an Image dropped on the bottom zone starts a parallel flow. A block
  dropped under a block starts a branch, and what comes before is shared and computed once.
- **The Output is not a block on screen.** It lives in the wire to Post; its settings are on the Post layer.

## What every action must respect

- **No blocking pop-ups** for ordinary actions: delete, remove a wire, remove a session. They happen at once, with a
  short message and Ctrl+Z to undo. Wires that cannot be removed are not selectable.
- **Every edit can be undone (Ctrl+Z) and redone (Ctrl+Y)**, back to exactly what was there. A session's frames are
  saved by Save, not undone, but removing or adding a Session block is undone.
- **A mistake is always correctable without throwing the project away.** Example: a file added by mistake can be
  taken out alone.
- **A file of a different type** than the others is warned about (on the Session block, in its panel, before a run).
- **Nothing runs without data.** Run says why when there are no frames and no image file. An empty workflow is not
  run, and nothing goes to Post from it.
- **The user's projects and files are never written** except by his own actions on them.

## Words and looks

- **Languages.** English interface, full French translation. In French, no English word may remain except
  "workflow", which stays in French too, and the standard names a stacking-software user knows.
- **Plain words.** No internal words: legacy, v1/v2, stage names such as render_simple or star_removal, keys such as
  "gradient~2" or "fond", file paths of the store, Python errors shown raw. Name things by what they do.
- **No external software name** in the interface: no stacking or editing program, no neural-network brand.
- **Self-explaining controls.** A control sits on the object it acts on, never "where there was room"; patterns follow
  the reference photo editors (panels, layers, menus, shortcuts). Text is not truncated without a tooltip. Nothing
  overlaps. A greyed thing says why. The scope of an action is visible: which block, which flow.
- **Screen size.** Readable at 1366×768 and at 1920×1080.
- **Consistency.** The same thing has the same name and the same behaviour everywhere: graph, palette, panel, menus,
  run window, Post, Inspect, the Log.

## Abilities that existed before (check they still work)

Branches; parallel flows; removable wires to Post; Split stars and its two images sent to Post; Combine stars &
background (a block in the Composition column); Fusion and its palettes; switch off and on; pause for review and mark
as reviewed; Run to here (Shift+F5); Inspect a block (double-click); rename a workflow or a session; save as template,
export and import templates (in the template window at the start); the GPU or CPU combo left of Run; the Log; the
language change at runtime; the health check of the frames; "Review frames…" opening Inspect; Ctrl+Z / Ctrl+Y; the
image source's kind (linear or stretched); the Session's filter.
