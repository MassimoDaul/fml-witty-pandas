(() => {
  const modeConfig = {
    papers: {
      placeholder: 'Try "machine learning for endangered languages"',
      hint: "Search by question, topic, method, field, or researcher.",
      landingSubmit: "Search papers",
      resultsSubmit: "Update results",
    },
    authors: {
      placeholder: 'Try "Geoffrey Hinton"',
      hint: "Search for a researcher by name. Paper filters stay secondary in this mode.",
      landingSubmit: "Find author",
      resultsSubmit: "Find author",
    },
  };

  const bookmarkStorageKey = "student-research-helper.bookmarks";

  function readBookmarks() {
    try {
      const raw = window.localStorage.getItem(bookmarkStorageKey);
      const parsed = raw ? JSON.parse(raw) : {};
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch {
      return {};
    }
  }

  function writeBookmarks(bookmarks) {
    try {
      window.localStorage.setItem(bookmarkStorageKey, JSON.stringify(bookmarks));
    } catch {
      // Ignore storage failures; bookmarking is an enhancement.
    }
  }

  function updateBookmarkButton(button, bookmarks) {
    const paperId = button.dataset.bookmarkId;
    if (!paperId) {
      return;
    }

    const saved = Boolean(bookmarks[paperId]);
    button.setAttribute("aria-pressed", saved ? "true" : "false");
    button.classList.toggle("is-saved", saved);

    const label = button.querySelector("span");
    if (label) {
      label.textContent = saved ? "Saved" : "Save";
    }
  }

  function hydrateBookmarks(root = document) {
    const bookmarks = readBookmarks();
    root.querySelectorAll("[data-bookmark-button]").forEach((button) => {
      updateBookmarkButton(button, bookmarks);
    });
  }

  function toggleBookmark(button) {
    const paperId = button.dataset.bookmarkId;
    const title = button.dataset.bookmarkTitle || "paper";
    if (!paperId) {
      return;
    }

    const bookmarks = readBookmarks();
    if (bookmarks[paperId]) {
      delete bookmarks[paperId];
    } else {
      bookmarks[paperId] = title;
    }

    writeBookmarks(bookmarks);
    updateBookmarkButton(button, bookmarks);
  }

  function getSelectedMode(form) {
    return form.querySelector('input[name="mode"]:checked')?.value || "papers";
  }

  function setSubmitting(form, submitting) {
    form.classList.toggle("is-submitting", submitting);
    const button = form.querySelector("[data-submit-button]");
    if (!button) {
      return;
    }

    const label = button.querySelector("[data-submit-label]");
    const mode = getSelectedMode(form);
    const isLanding = form.classList.contains("search-form--landing");
    const config = modeConfig[mode];

    button.disabled = submitting;
    if (label) {
      if (submitting) {
        label.textContent = "Searching...";
      } else {
        label.textContent = isLanding ? config.landingSubmit : config.resultsSubmit;
      }
    }
  }

  function updateMode(form) {
    const mode = getSelectedMode(form);
    const config = modeConfig[mode];
    const input = form.querySelector("[data-search-input]");
    const hint = form.querySelector("[data-search-hint]");
    const submitLabel = form.querySelector("[data-submit-label]");
    const isLanding = form.classList.contains("search-form--landing");

    if (input) {
      input.placeholder = config.placeholder;
    }

    if (hint) {
      hint.textContent = config.hint;
    }

    if (submitLabel && !form.classList.contains("is-submitting")) {
      submitLabel.textContent = isLanding ? config.landingSubmit : config.resultsSubmit;
    }

    const paperFilterBlock = document.querySelector("[data-paper-filters]");
    const authorNote = document.querySelector("[data-author-note]");
    const sortControl = document.querySelector("[data-paper-sort]");
    const categoriesInput = document.querySelector('input[name="categories"]');
    const yearInputs = document.querySelectorAll('input[name="year_from"], input[name="year_to"]');
    const sortSelect = sortControl?.querySelector("select");
    const paperCountSelect = document.getElementById("results-k");
    const authorCountSelect = document.getElementById("results-k-authors");

    if (paperFilterBlock) {
      const showPaperFilters = mode === "papers";
      paperFilterBlock.hidden = !showPaperFilters;
      if (categoriesInput) {
        categoriesInput.disabled = !showPaperFilters;
      }
      yearInputs.forEach((field) => {
        field.disabled = !showPaperFilters;
      });
    }

    if (paperCountSelect) {
      paperCountSelect.disabled = mode !== "papers";
    }

    if (authorCountSelect) {
      authorCountSelect.disabled = mode !== "authors";
    }

    if (authorNote) {
      authorNote.hidden = mode !== "authors";
    }

    if (sortControl && sortSelect) {
      const enableSort = mode === "papers";
      sortControl.classList.toggle("is-disabled", !enableSort);
      sortControl.setAttribute("aria-disabled", enableSort ? "false" : "true");
      sortSelect.disabled = !enableSort;
    }
  }

  function initSearchForm(form) {
    form.querySelectorAll('input[name="mode"]').forEach((input) => {
      input.addEventListener("change", () => updateMode(form));
    });

    form.addEventListener("submit", () => {
      setSubmitting(form, true);
    });

    updateMode(form);
  }

  function openFilters() {
    document.body.classList.add("filters-open");
  }

  function closeFilters() {
    document.body.classList.remove("filters-open");
  }

  document.querySelectorAll("[data-search-form]").forEach(initSearchForm);
  hydrateBookmarks();

  document.addEventListener("click", (event) => {
    const bookmarkButton = event.target.closest("[data-bookmark-button]");
    if (bookmarkButton instanceof HTMLElement) {
      event.preventDefault();
      toggleBookmark(bookmarkButton);
      return;
    }

    const openTrigger = event.target.closest("[data-filters-open]");
    if (openTrigger instanceof HTMLElement) {
      event.preventDefault();
      openFilters();
      return;
    }

    const closeTrigger = event.target.closest("[data-filters-close]");
    if (closeTrigger instanceof HTMLElement) {
      event.preventDefault();
      closeFilters();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeFilters();
    }
  });

  if (window.htmx) {
    document.body.addEventListener("htmx:beforeRequest", (event) => {
      const form = event.detail?.elt?.closest?.("[data-search-form]");
      if (form instanceof HTMLElement) {
        setSubmitting(form, true);
      }
    });

    document.body.addEventListener("htmx:afterSwap", (event) => {
      if (event.target instanceof HTMLElement && event.target.id === "results-content") {
        closeFilters();
        hydrateBookmarks(event.target);
      }

      const form = document.querySelector("[data-search-form]");
      if (form instanceof HTMLElement) {
        setSubmitting(form, false);
        updateMode(form);
      }
    });

    document.body.addEventListener("htmx:responseError", () => {
      const form = document.querySelector("[data-search-form]");
      if (form instanceof HTMLElement) {
        setSubmitting(form, false);
      }
    });
  }

  window.addEventListener("pageshow", () => {
    document.querySelectorAll("[data-search-form]").forEach((form) => {
      setSubmitting(form, false);
      updateMode(form);
    });
    hydrateBookmarks();
  });
})();
