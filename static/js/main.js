/**
 * main.js – Credit Card Approval Prediction
 * Minor UX enhancements
 */

document.addEventListener("DOMContentLoaded", () => {
  // Animate confidence bar on result page
  const bar = document.querySelector(".confidence-bar");
  if (bar) {
    const target = bar.style.width;
    bar.style.width = "0%";
    setTimeout(() => { bar.style.width = target; }, 200);
  }

  // Highlight active nav link
  const links = document.querySelectorAll(".nav-links a");
  links.forEach(link => {
    if (link.href === window.location.href) {
      link.style.color = "#e2b96f";
      link.style.fontWeight = "700";
    }
  });
});
