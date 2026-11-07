"""post_wires B: a middle block's image sent to Post (menu, port pulled), twice; then the branch's wire removed, the
original block deleted; what the Post block says and its panel lists; Ctrl+Z / Ctrl+Y."""
import sys

from explore_post_common import CHAINS, FR, VIEWKEY, P, Explorer, build_chain, session_project, start, steps_chain
from PySide6.QtCore import Qt

start(2400)
ex = Explorer("post_wires")
pp = P(ex)
mids = sys.argv[1:] or ["stacking", "astrometry", "gradient", "denoise", "color"]


def consistency(tag, st):
    """The layer count on the Post block, in its panel and in the graph's list."""
    n_graph = len(pp.graph_layers())
    summ = pp.final_summary()
    ex.click_block(pp.final())
    rows = [t for t in pp.panel_texts() if "<table" in t]
    n_rows = rows[0].count("<tr>") if rows else 0
    num = int(summ.split()[0]) if summ.split() and summ.split()[0].isdigit() else -1
    wires = len(pp.wires()) + len(pp.post_arrows())
    print(f"  [{tag}] summary={summ!r} graph={n_graph} panel_rows={n_rows} wires={len(pp.wires())} "
          f"arrows={len(pp.post_arrows())} comps={len(pp.blocks_of('composition'))}")
    if num != n_rows or num != n_graph:
        pp.note_once(f"count_{tag}", "gênant", "Nombre de calques incohérent entre le bloc Post-traitement et son panneau",
                     f"Le bloc dit « {summ} », son panneau liste {n_rows} calque(s), le graphe en compte {n_graph}.", st)
    return num


for mid in mids:
    for how in ("menu", "port"):
        print("=" * 30, mid, how)
        end = "stars" if mid == "color" else "color"
        session_project(ex)
        build_chain(ex, CHAINS[end])
        st = steps_chain(end)
        k = VIEWKEY[mid]
        n0 = consistency("start", st)
        it = pp.block(k)
        npaths0 = len(ex.s.paths)
        act = f"Clic droit sur {FR[mid]} > Envoyer son image au Post" if how == "menu" else \
            f"Tirer le port ● de {FR[mid]} sur le bloc Post-traitement"
        if how == "menu":
            m = ex.right_click_block(it)
            texts = [d.get("text") for d in m]
            if not any("Vers Post" in (t or "") for t in texts):
                pp.note_once(f"nomenu_{mid}", "gênant", f"{FR[mid]} : pas d'entrée « Envoyer son image au Post »",
                             f"Menu : {texts}", st + [f"Clic droit sur {FR[mid]}"])
                continue
            ex.right_click_block(pp.block(k), "Vers Post")
        else:
            if not it.ports:
                pp.note_once(f"noport_{mid}", "gênant", f"{FR[mid]} n'a pas de port ● vers le Post-traitement", "", st)
                continue
            pp.pull_port(it)
        ex.settle(6)
        pp.check_slot_errors(act, st + [act])
        pp.dump(f"after send {mid} by {how}")
        toast = pp.toast()
        print("toast:", toast)
        pp.check_texts("message", [toast], st + [act])
        import re as _re
        if _re.search(r"\bde [AEÉIOU]", toast):
            pp.note_once("elision", "mineur", "Français : « sortie de Astrométrie » au lieu de « sortie d'Astrométrie »",
                         f"Message : « {toast} » ; le calque s'appelle « {ex.s.paths[-1].name} ». Même chose pour "
                         "Empilement (« sortie de Empilement »).", st + [act])
        if len(ex.s.paths) == npaths0:
            pp.note_once(f"nosend_{mid}_{how}", "gênant", f"{act} : rien ne se passe", f"Message : « {toast} »", st + [act])
            continue
        q = ex.s.paths[-1]
        print("new path:", q.pid, q.name, q.end, ex.s.gives_post(q))
        pp.check_texts("nom du nouveau calque", [q.name], st + [act])
        n1 = consistency(f"after_send_{mid}_{how}", st + [act])
        if n1 != n0 + 1:
            pp.note_once(f"count_after_send_{mid}", "gênant", f"{act} : le Post-traitement ne gagne pas un calque",
                         f"Avant : {n0} calque(s), après : {n1}.", st + [act])
        # selection and panel: what is shown after the send
        print("sel after send:", ex.s.sel_step, ex.s.path.name if ex.s.path else None, pp.panel_texts()[:3])
        ex.shot(f"send_{mid}_{how}")
        # twice
        it = pp.block(k, pid=ex.s.paths[0].pid)
        m = ex.right_click_block(it)
        texts = [d.get("text") for d in m]
        print("menu after send:", texts)
        npaths1 = len(ex.s.paths)
        if how == "menu":
            if any("Vers Post" in (t or "") for t in texts):
                ex.right_click_block(it, "Vers Post")
        else:
            if it.ports:
                pp.pull_port(it)
        ex.settle(6)
        pp.check_slot_errors(act + " (deuxième fois)", st + [act, act])
        print("twice: paths", npaths1, "->", len(ex.s.paths), "toast:", pp.toast())
        if len(ex.s.paths) > npaths1:
            pp.note_once(f"twice_{mid}", "mineur", f"{FR[mid]} envoyé deux fois au Post-traitement : deux calques "
                         "identiques", f"Après la 2e fois : {[p.name for p in ex.s.paths]}", st + [act, act])
        elif not pp.toast() and how == "port":
            pp.note_once(f"twice_silent_{mid}", "mineur", f"Deuxième envoi de {FR[mid]} au Post-traitement : rien ne "
                         "se passe, sans message", "Le port tiré à nouveau ne fait rien et ne dit pas que l'image est "
                         "déjà un calque.", st + [act, act])
        # Ctrl+Z / Ctrl+Y of the send
        pp.undo()
        print("undo: paths", [p.name for p in ex.s.paths], "summary", pp.final_summary(), "toast", pp.toast())
        n_u = consistency(f"undo_{mid}_{how}", st + [act, "Ctrl+Z"])
        if len(ex.s.paths) != npaths0:
            pp.note_once(f"undo_send_{mid}", "gênant", f"Ctrl+Z après {act} ne retire pas le calque", "", st + [act])
        pp.redo()
        n_r = consistency(f"redo_{mid}_{how}", st + [act, "Ctrl+Z", "Ctrl+Y"])
        print("redo: paths", [p.name for p in ex.s.paths], "summary", pp.final_summary())
        if len(ex.s.paths) != npaths1:
            pp.note_once(f"redo_send_{mid}", "gênant", f"Ctrl+Y après Ctrl+Z ne remet pas le calque envoyé ({FR[mid]})",
                         f"Workflows : {[p.name for p in ex.s.paths]}", st + [act, "Ctrl+Z", "Ctrl+Y"])
        if how == "menu":
            # remove the branch's wire to Post
            br = ex.s.paths[-1]
            wl = [w for w in pp.wires() if br.name in w.toolTip()]
            print("branch wires:", [w.toolTip() for w in pp.wires()], "->", len(wl))
            if wl and pp.click_wire(wl[0]):
                pp.del_key()
                pp.check_slot_errors("Suppr sur le fil de la branche envoyée", st + [act])
                pp.dump("after branch wire removed")
                print("toast:", pp.toast())
                still = [p for p in ex.s.paths if p.pid == br.pid]
                bl = [b for b in ex.blocks() if any(q.pid == br.pid for q in b.node.paths)]
                print("branch still there:", bool(still), "its blocks:", [(b.node.key, len(b.node.paths)) for b in bl])
                if still:
                    pp.note_once("branch_left", "question",
                                 "Fil d'une image envoyée au Post retiré : la branche reste dans le graphe",
                                 f"Après {act} puis Suppr sur son fil, le calque part mais le workflow « {br.name} » reste "
                                 "(sans fil vers le Post). Il compte encore dans l'exécution.",
                                 st + [act, "Clic sur le fil de la nouvelle branche", "Suppr"])
                consistency(f"branchwire_{mid}", st + [act, "Suppr sur son fil"])
                pp.undo()
                print("undo branch wire removal:", [p.name for p in ex.s.paths], len(pp.wires()))
        else:
            # delete the original block from the main workflow
            main = ex.s.paths[0]
            it = pp.block(k, pid=main.pid)
            m = ex.right_click_block(it)
            texts = [d.get("text") for d in m]
            print("menu of shared block:", texts)
            pick = next((t for t in texts if t and ("Supprimer" in t or "Retirer" in t)), None)
            if pick:
                ex.right_click_block(it, pick.split("\t")[0])
                pp.check_slot_errors(f"suppression de {FR[mid]} après l'envoi", st + [act])
                pp.dump(f"after deleting {mid}")
                toast = pp.toast()
                print("toast:", toast)
                st_del = st + [act, f"Clic droit sur {FR[mid]} > « {pick.split(chr(9))[0]} »"]
                br = ex.s.paths[-1]
                bblocks = [n["block"] for n in ex.s.p.workflow(br.pid).blocks]
                mblocks = [n["block"] for n in ex.s.p.workflow(main.pid).blocks]
                if mid in mblocks and mid not in bblocks:
                    pp.note_once("partial_remove", "gênant", "« Retirer des 2 workflows » : refusé dans l'un, fait "
                                 "dans l'autre", f"Le message dit seulement « {toast} », mais {FR[mid]} a quand même été "
                                 f"retiré de « {br.name} » (qui devient {bblocks}). Le refus n'est pas tout ou rien.",
                                 st_del)
                if mid not in bblocks and FR[mid] in br.name and len(ex.s.paths) > 1:
                    pp.note_once("stale_name", "gênant", "Le calque envoyé garde le nom d'un bloc qui n'est "
                                 "plus là", f"Après la suppression de {FR[mid]}, la branche « {br.name} » donne l'image "
                                 f"de « {bblocks[-2] if len(bblocks) > 1 else '-'} » mais garde son nom ; le bloc "
                                 f"Post-traitement liste toujours « {br.name} ».", st_del)
                consistency(f"delorig_{mid}", st + [act, f"Supprimer {FR[mid]}"])
                pp.undo()
                pp.dump("undo delete")
                consistency(f"delorig_undo_{mid}", st + [act, f"Supprimer {FR[mid]}", "Ctrl+Z"])
ex.close()
