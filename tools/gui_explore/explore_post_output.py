"""post_wires G: the Output block drawn for a branch sent to Post (it should live in the wire): its menu, Del, cross,
its panel; then the English names of the layers (lang en)."""
import sys

from explore_post_common import CHAINS, P, Explorer, build_chain, session_project, start, steps_chain

start(1200)
lang = sys.argv[1] if len(sys.argv) > 1 else "fr"
ex = Explorer("post_wires", lang=lang)
pp = P(ex)

session_project(ex)
build_chain(ex, CHAINS["color"])
st = steps_chain("color") + ["Clic droit sur Gradient > Envoyer son image au Post"]
ex.right_click_block(pp.block("fond"), "Vers Post" if lang == "fr" else "To Post")
po = [b for b in ex.blocks() if b.node.key == "post"]
print("visible Output blocks:", [(b.title, b.summary) for b in po])
if po and lang == "fr":
    b = po[0]
    pp.note_once("output_visible", "gênant", "Un bloc « Sortie » apparaît dans le graphe après un envoi au Post",
                 f"Après « Envoyer son image au Post » sur Gradient, la branche créée montre un bloc « {b.title} » "
                 f"(« {b.summary} ») entre Gradient et Post-traitement. La règle : la Sortie n'est pas un bloc à "
                 "l'écran, elle vit dans le fil vers le Post.", st)
    m = ex.right_click_block(b)
    texts = [d.get("text") for d in m]
    print("Output menu:", texts)
    odd = [t for t in texts if t and "absent du workflow" in t]
    if odd:
        pp.note_once("output_menu_odd", "mineur", "Menu du bloc Sortie : « Désactiver — ce bloc n'est pas dans le "
                     "workflow »", f"Menu complet : {texts}. Le bloc est pourtant dans la branche ; et « Supprimer le "
                     "workflow » et « Supprimer le bloc » se suivent sans séparateur.", st + ["Clic droit sur Sortie"])
    ex.click_block(b)
    print("Output panel:", pp.panel_texts(), pp.panel_buttons())
    # Del on the Output block
    pp.del_key()
    pp.check_slot_errors("Suppr sur le bloc Sortie", st)
    print("after Del on Output: paths", [q.name for q in ex.s.paths], "toast", pp.toast(), "popups", ex.popups[-2:])
    po2 = [x for x in ex.blocks() if x.node.key == "post"]
    if po2 and not pp.toast():
        pp.note_once("output_del_silent", "mineur", "Suppr sur le bloc Sortie : rien, sans message", "", st)
    elif po2 and "absent du workflow" in pp.toast():
        pp.note_once("output_del_msg", "mineur", "Suppr sur le bloc Sortie : message incompréhensible",
                     f"Le menu offre « Supprimer le bloc  Suppr », mais Suppr répond « {pp.toast()} » : le bloc est "
                     "nommé comme le workflow et la raison est fausse (il est bien dans le workflow).",
                     st + ["Clic sur le bloc Sortie", "Suppr"])
    pt = pp.panel_texts()
    if "Canaux liés" in pt:
        pp.note_once("channels_two_controls", "mineur", "Le réglage des canaux de la Sortie n'a pas le même nom ni la "
                     "même commande dans le graphe et dans le Post-traitement",
                     "Panneau du bloc Sortie : une case « Canaux liés » (groupe « Étirement ») ; page Post-traitement, "
                     "calque du workflow : une liste « Canaux : Liés / Chaque canal ». C'est le même réglage.",
                     st + ["Clic sur le bloc Sortie"])
    # its menu's Delete block
    if po2:
        ex.right_click_block(po2[0], "Supprimer")
        pp.check_slot_errors("Supprimer le bloc Sortie (menu)", st)
        print("after menu delete: paths", [q.name for q in ex.s.paths], "toast", pp.toast(), "popups", ex.popups[-2:])
        po3 = [x for x in ex.blocks() if x.node.key == "post"]
        print("output still:", bool(po3))
    # "Supprimer le workflow" from its menu: a blocking question?
    po3 = [x for x in ex.blocks() if x.node.key == "post"]
    if po3:
        n = len(ex.popups)
        ex.right_click_block(po3[0], "Supprimer workflow")
        print("delete workflow popups:", ex.popups[n:], "paths", [q.name for q in ex.s.paths])
        if len(ex.popups) > n:
            pp.note_once("delete_wf_popup", "mineur", "« Supprimer le workflow » demande une confirmation bloquante",
                         f"Question : {ex.popups[n:]}. Les autres suppressions se font tout de suite avec Ctrl+Z.",
                         st + ["Clic droit sur Sortie > Supprimer le workflow"])
# names in the language
ex.right_click_block(pp.block("etoiles") or pp.block("couleur"), None)
session_project(ex)
build_chain(ex, CHAINS["stars"])
pp.pull_port(pp.block("etoiles"), "starless")
pp.pull_port(pp.block("etoiles"), "stars")
pp.pull_port(pp.block("couleur"))
ex.click_block(pp.final())
print(lang, "post panel:", pp.panel_texts())
print(lang, "graph layers:", pp.graph_layers())
print(lang, "wire tips:", [w.toolTip() for w in pp.wires()])
print(lang, "split tip:", pp.block("etoiles").toolTip())
print(lang, "split menu:", [d.get("text") for d in ex.right_click_block(pp.block("etoiles"))])
print(lang, "toast:", pp.toast())
if lang == "fr":
    words = set()
    texts = [w.toolTip() for w in pp.wires()] + [d.get("text") or "" for d in ex.menus[-1]] + \
        [pp.block("etoiles").toolTip(), pp.toast()] + pp.panel_texts()
    ex.click_block(pp.block("etoiles"))
    texts += pp.panel_texts()
    import re
    for t in texts:
        for m in re.finditer(r"(au Post\b|du Post\b|le Post\b|à Post-traitement|en Post-traitement|vers Post-traitement|"
                             r"vers le Post\b|de Post-traitement|dans le Post\b)", t):
            words.add(m.group(0))
    print("Post words:", words)
    if len(words) > 2:
        pp.note_once("post_word_mix", "mineur", "Le même bloc s'appelle tantôt « Post », tantôt « Post-traitement »",
                     "Le bloc et le bouton disent « Post-traitement », mais les menus, infobulles et messages mélangent : "
                     + ", ".join(f"« {w} »" for w in sorted(words)) + " (ex. menu de Split : « Envoyer son image au Post » "
                     "puis « Envoyer l'image sans étoiles à Post-traitement » puis « Retirer … du Post »).",
                     steps_chain("stars") + ["Clic droit sur Split étoiles et fond"])
ex.close()
