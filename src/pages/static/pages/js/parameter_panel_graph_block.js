/*
 * Admin form of a graph in a parameter panel (pages.blocks.ParameterPanelGraphBlock).
 *
 * The block's choices list every dimension and category of the instance, because they are fixed
 * before the editor picks a node. This script narrows them to the chosen node: it asks the server
 * for the node's dimensions and the categories it has data for, and hides the rest.
 *
 * - When the editor picks another node, choices that no longer apply are cleared.
 * - When the form is opened, saved choices are left alone, even if they no longer apply, so that
 *   the editor sees them (and the server's validation error) instead of losing them silently.
 * - If the server cannot answer, everything stays visible, so the editor is never blocked.
 */
class ParameterPanelGraphBlockDefinition extends window.wagtailStreamField.blocks
  .StructBlockDefinition {
  render(placeholder, prefix, initialState, initialError) {
    const block = super.render(placeholder, prefix, initialState, initialError);
    const container = block.container[0];
    const { nodeOptionsUrl, texts } = this.meta;

    const nodeInput = document.getElementById(`${prefix}-node`);
    const groupBySelect = document.getElementById(`${prefix}-group_by`);
    const filtersField = container.querySelector('[data-contentpath="filters"]');
    const filterInputs = () =>
      Array.from(container.querySelectorAll(`input[name="${prefix}-filters"]`));

    // A hint above the categories, for when there is nothing to choose.
    const hint = document.createElement('p');
    hint.className = 'help';
    hint.hidden = true;
    if (filtersField) filtersField.prepend(hint);
    const showHint = (text) => {
      hint.textContent = text || '';
      hint.hidden = !text;
    };

    // `allowed`: Map of dimension id -> Set of category ids, or null to show everything.
    const apply = (allowed, clearInvalid) => {
      if (groupBySelect) {
        for (const option of groupBySelect.options) {
          if (!option.value) continue;
          const ok = allowed === null || allowed.has(option.value);
          if (!ok && clearInvalid && option.selected) groupBySelect.value = '';
          option.hidden = !ok && !option.selected;
          option.disabled = !ok && !option.selected;
        }
      }

      const groups = new Set();
      for (const input of filterInputs()) {
        const [dimId, catId] = input.value.split(':');
        const ok = allowed === null || (allowed.has(dimId) && allowed.get(dimId).has(catId));
        if (!ok && clearInvalid) input.checked = false;
        // Django renders each checkbox in its own <div>, inside a <div> per dimension.
        const wrapper = input.closest('div');
        if (!wrapper) continue;
        wrapper.hidden = !ok && !input.checked;
        if (wrapper.parentElement && wrapper.parentElement !== filtersField) {
          groups.add(wrapper.parentElement);
        }
      }
      for (const group of groups) {
        const options = Array.from(group.children).filter((el) => el.tagName === 'DIV');
        if (!options.length) continue;
        group.hidden = options.every((el) => el.hidden);
      }
    };

    let requestId = 0;
    const refresh = async (clearInvalid) => {
      const nodeId = nodeInput ? nodeInput.value : '';
      const thisRequest = ++requestId;
      if (!nodeId) {
        apply(new Map(), clearInvalid);
        showHint(texts.chooseNode);
        return;
      }
      let data;
      try {
        const url = `${nodeOptionsUrl}?node=${encodeURIComponent(nodeId)}`;
        const response = await fetch(url, {
          credentials: 'same-origin',
          headers: { Accept: 'application/json' },
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        data = await response.json();
      } catch (err) {
        console.warn('Could not fetch the options of node', nodeId, err);
        if (thisRequest === requestId) {
          apply(null, false);
          showHint('');
        }
        return;
      }
      // A newer request (the editor picked another node meanwhile) wins.
      if (thisRequest !== requestId) return;
      const allowed = new Map(
        data.dimensions.map((dim) => [dim.id, new Set(dim.categories.map((cat) => cat.id))]),
      );
      apply(allowed, clearInvalid);
      showHint(allowed.size ? '' : texts.noDimensions);
    };

    void refresh(false);
    if (nodeInput) {
      // The node chooser announces a new choice with a jQuery event, which a native listener
      // would not see.
      const onChange = () => void refresh(true);
      if (window.jQuery) window.jQuery(nodeInput).on('change', onChange);
      else nodeInput.addEventListener('change', onChange);
    }

    return block;
  }
}

window.telepath.register('paths.pages.ParameterPanelGraphBlock', ParameterPanelGraphBlockDefinition);
