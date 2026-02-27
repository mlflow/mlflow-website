import { useState, useLayoutEffect, useEffect } from "react";
import Link from "@docusaurus/Link";
import { useLocation } from "@docusaurus/router";
import useBaseUrl from "@docusaurus/useBaseUrl";
import Logo from "@site/static/img/mlflow-logo-white.svg";
import DownIcon from "@site/static/img/chevron-down-small.svg";

import { cn, getStartedLinkForPage } from "../../utils";
import { useGitHubStars } from "../../hooks/useGitHubStars";

import { Star } from "lucide-react";
import { Button } from "../Button/Button";
import { HeaderMenuItem } from "../HeaderMenuItem/HeaderMenuItem";
import { HeaderProductsSubmenu } from "../HeaderProductsSubmenu/HeaderProductsSubmenu";
import { HeaderDocsSubmenu } from "../HeaderDocsSubmenu/HeaderDocsSubmenu";

const GitHubStarsBadge = () => {
  const stars = useGitHubStars();
  return (
    <a
      href="https://github.com/mlflow/mlflow"
      target="_blank"
      rel="noreferrer noopener"
      className="hidden md:flex items-center gap-1.5 rounded-xl bg-white/10 border border-white/20 px-4 py-2 text-[15px] text-white hover:bg-white/15 hover:border-white/30 transition-all"
    >
      <svg viewBox="0 0 48 48" fill="none" className="w-4 h-4">
        <path
          d="M24 3.9C35.05 3.9 44 12.85 44 23.9c0 4.19-1.315 8.275-3.759 11.68-2.444 3.404-5.894 5.955-9.864 7.296-.996.2-1.371-.425-1.371-.95 0-.675.025-2.825.025-5.5 0-1.875-.625-3.075-1.35-3.7 4.45-.5 9.125-2.2 9.125-9.875 0-2.2-.775-3.975-2.05-5.375.2-.5.9-2.55-.2-5.3 0 0-1.675-.55-5.5 2.05-1.6-.45-3.3-.675-5-.675s-3.7.225-5.3.675c-3.825-2.575-5.5-2.05-5.5-2.05-1.1 2.75-.4 4.8-.2 5.3-1.275 1.4-2.05 3.2-2.05 5.375 0 7.65 4.65 9.4 9.1 9.9-.575.5-1.1 1.375-1.275 2.675-1.15.525-4.025 1.375-5.825-1.65-.375-.6-1.5-2.075-3.075-2.05-1.675.025-.675.95.025 1.325.85.475 1.825 2.25 2.05 2.825.4 1.125 1.7 3.275 6.725 2.35 0 1.675.025 3.25.025 3.725 0 .525-.375 1.125-1.375.95-3.983-1.326-7.447-3.872-9.902-7.278C5.318 32.192 3.998 28.1 4 23.901 4 12.851 12.95 3.9 24 3.9Z"
          fill="currentColor"
        />
      </svg>
      <Star className="w-3 h-3" fill="currentColor" />
      {stars && <span className="font-medium">{stars}</span>}
    </a>
  );
};

import "./Header.module.css";
import { MLFLOW_DOCS_URL } from "@site/src/constants";
import { cva } from "class-variance-authority";

const MD_BREAKPOINT = 640;

const navStyles = cva(
  "fixed w-full z-20 top-0 start-0 bg-black/20 border-b border-[#F7F8F8]/8 backdrop-blur-[20px] overflow-y-auto",
  {
    variants: {
      isOpen: {
        true: "h-dvh",
        false: "",
      },
    },
  },
);

export const Header = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isProductItemHovered, setIsProductItemHovered] = useState(false);
  const [isProductSubmenuOpen, setIsProductSubmenuOpen] = useState(false);
  const [isDocsItemHovered, setIsDocsItemHovered] = useState(false);
  const [isDocsSubmenuOpen, setIsDocsSubmenuOpen] = useState(false);
  const location = useLocation();
  const classicalMLPath = useBaseUrl("/classical-ml");
  const genAIPath = useBaseUrl("/genai");

  const getStartedHref = getStartedLinkForPage(
    location.pathname,
    classicalMLPath,
    genAIPath,
  );

  const handleProductItemHover = () => {
    setIsProductItemHovered(true);
    setIsDocsItemHovered(false);
  };

  const handleProductSubmenuLeave = () => {
    setIsProductItemHovered(false);
  };

  const toggleProductSubmenuHovered = () => {
    setIsProductItemHovered(!isProductItemHovered);
  };

  const handleProductItemClick = () => {
    const newState = !isProductSubmenuOpen;
    setIsProductSubmenuOpen(newState);
    // Accordion behavior: close Docs when Components opens
    if (newState) {
      setIsDocsSubmenuOpen(false);
    }
  };

  const handleDocsItemHover = () => {
    setIsDocsItemHovered(true);
    setIsProductItemHovered(false);
  };

  const handleDocsSubmenuLeave = () => {
    setIsDocsItemHovered(false);
  };

  const toggleDocsSubmenuHovered = () => {
    setIsDocsItemHovered(!isDocsItemHovered);
  };

  const handleDocsItemClick = () => {
    const newState = !isDocsSubmenuOpen;
    setIsDocsSubmenuOpen(newState);
    // Accordion behavior: close Components when Docs opens
    if (newState) {
      setIsProductSubmenuOpen(false);
    }
  };

  useLayoutEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= MD_BREAKPOINT) {
        setIsOpen(false);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    if (isOpen) {
      document.body.classList.add("noScroll");
    } else {
      document.body.classList.remove("noScroll");
      // Reset submenus when menu closes
      setIsProductSubmenuOpen(false);
      setIsDocsSubmenuOpen(false);
    }
  }, [isOpen]);

  return (
    <nav className={navStyles({ isOpen })}>
      <div className="flex flex-wrap items-center mx-auto px-6 lg:px-20 py-2 max-w-container">
        <div className="md:contents flex flex-row justify-between w-full sticky top-[8px]">
          <Link
            href="/"
            className="flex items-center space-x-3 rtl:space-x-reverse grow basis-0"
            aria-label="MLflow Home"
          >
            <Logo className="h-[36px]" aria-hidden="true" />
          </Link>
          <div className="flex flex-row items-center gap-3 md:order-2 md:gap-4 rtl:space-x-reverse grow justify-end basis-0">
            <GitHubStarsBadge />
            <Link href={getStartedHref} className="hidden md:block">
              <Button variant="primary" size="small">
                Get started
              </Button>
            </Link>
            <button
              data-collapse-toggle="navbar-sticky"
              type="button"
              className="inline-flex items-center p-2 w-10 h-10 justify-center text-sm text-white md:hidden focus:outline-none cursor-pointer"
              onClick={() => setIsOpen(!isOpen)}
            >
              <span className="sr-only">Open main menu</span>
              <svg
                className="w-5 h-5"
                aria-hidden="true"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 17 14"
              >
                <path
                  stroke="currentColor"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M1 1h15M1 7h15M1 13h15"
                />
              </svg>
            </button>
          </div>
        </div>
        <div
          className={cn(
            "items-center justify-between w-full md:flex md:w-auto md:order-1 mt-4 md:mt-0",
            isOpen ? "flex" : "hidden md:flex",
          )}
        >
          <ul className="flex flex-col font-medium md:flex-row gap-y-6 gap-x-4 lg:gap-x-10 w-full md:w-auto">
            <li className="w-full md:w-auto md:hidden">
              <button
                type="button"
                onClick={handleProductItemClick}
                aria-expanded={isProductSubmenuOpen}
                aria-controls="mobile-components-submenu"
                className="flex items-center justify-between gap-2 py-2 text-white text-lg w-full cursor-pointer transition-colors duration-200 hover:text-white/60"
              >
                Components
                <DownIcon
                  className={cn(
                    "w-6 h-6 transition-transform duration-300",
                    isProductSubmenuOpen && "rotate-180",
                  )}
                />
              </button>
              <div
                id="mobile-components-submenu"
                className={cn(
                  "transition-all duration-300 ease-in-out overflow-hidden",
                  isProductSubmenuOpen
                    ? "max-h-[600px] opacity-100"
                    : "max-h-0 opacity-0",
                )}
              >
                <HeaderProductsSubmenu />
              </div>
            </li>
            <li
              className="w-full md:w-auto hidden md:block"
              onMouseEnter={handleProductItemHover}
              onClick={toggleProductSubmenuHovered}
            >
              <HeaderMenuItem label="Components" hasDropdown />
            </li>
            <li className="w-full md:w-auto">
              <HeaderMenuItem href="/releases" label="Releases" />
            </li>
            <li className="w-full md:w-auto">
              <HeaderMenuItem href="/blog" label="Blog" />
            </li>
            <li className="w-full md:w-auto md:hidden">
              <button
                type="button"
                onClick={handleDocsItemClick}
                aria-expanded={isDocsSubmenuOpen}
                aria-controls="mobile-docs-submenu"
                className="flex items-center justify-between gap-2 py-2 text-white text-lg w-full cursor-pointer transition-colors duration-200 hover:text-white/60"
              >
                Docs
                <DownIcon
                  className={cn(
                    "w-6 h-6 transition-transform duration-300",
                    isDocsSubmenuOpen && "rotate-180",
                  )}
                />
              </button>
              <div
                id="mobile-docs-submenu"
                className={cn(
                  "transition-all duration-300 ease-in-out overflow-hidden",
                  isDocsSubmenuOpen
                    ? "max-h-[300px] opacity-100"
                    : "max-h-0 opacity-0",
                )}
              >
                <HeaderDocsSubmenu />
              </div>
            </li>
            <li
              className="w-full md:w-auto hidden md:block"
              onMouseEnter={handleDocsItemHover}
              onClick={toggleDocsSubmenuHovered}
            >
              <HeaderMenuItem label="Docs" hasDropdown />
            </li>
            <li className="w-full md:w-auto">
              <HeaderMenuItem href="/ambassadors" label="Ambassador Program" />
            </li>
            <li className="w-full md:w-auto md:hidden">
              <Link href={getStartedHref}>
                <Button variant="primary" size="small" width="full">
                  Get started
                </Button>
              </Link>
            </li>
          </ul>
        </div>
      </div>

      <div
        className={cn(
          "flex-col w-full py-6",
          isProductItemHovered ? "flex" : "hidden",
        )}
        onMouseLeave={handleProductSubmenuLeave}
      >
        <HeaderProductsSubmenu />
      </div>

      <div
        className={cn(
          "flex-col w-full py-6",
          isDocsItemHovered ? "flex" : "hidden",
        )}
        onMouseLeave={handleDocsSubmenuLeave}
      >
        <HeaderDocsSubmenu />
      </div>
    </nav>
  );
};
