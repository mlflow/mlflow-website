import { useState, useLayoutEffect } from "react";
import Link from "@docusaurus/Link";
import Logo from "@site/static/img/mlflow-logo-white.svg";
import DownIcon from "@site/static/img/chevron-down-small.svg";

import { cn } from "../../utils";

import { Button } from "../Button/Button";
import { HeaderMenuItem } from "../HeaderMenuItem/HeaderMenuItem";
import { HeaderProductsSubmenu } from "../HeaderProductsSubmenu/HeaderProductsSubmenu";

import "./Header.module.css";
import { MLFLOW_DOCS_URL, MLFLOW_DBX_TRIAL_URL } from "@site/src/constants";

const MD_BREAKPOINT = 640;

type Props = {
  layoutType:
    | "default"
    | "genai"
    | "genai-subpage"
    | "classical-ml"
    | "classical-ml-subpage"
    | "home";
};

export const Header = ({ layoutType }: Props) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isProductItemHovered, setIsProductItemHovered] = useState(false);
  const [isProductSubmenuOpen, setIsProductSubmenuOpen] = useState(false);

  const handleProductItemHover = () => {
    setIsProductItemHovered(true);
  };

  const handleProductSubmenuLeave = () => {
    setIsProductItemHovered(false);
  };

  const toggleProductSubmenuHovered = () => {
    setIsProductItemHovered(!isProductItemHovered);
  };

  const handleProductItemClick = () => {
    setIsProductSubmenuOpen(!isProductSubmenuOpen);
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

  return (
    <nav className="fixed w-full z-20 top-0 start-0 bg-[#F7F8F8]/1 border-b border-[#F7F8F8]/8 backdrop-blur-[20px] drop-shadow-[0px_1px_2px_rgba(0_0_0/75%),0px_1px_12px_rgba(0_0_0/75%)]">
      <div className="flex flex-wrap items-center mx-auto px-6 lg:px-20 py-2 max-w-container">
        <Link
          href="/"
          className="flex items-center space-x-3 rtl:space-x-reverse grow basis-0"
        >
          <Logo className="h-[36px]" />
        </Link>
        <div className="flex flex-row items-center gap-6 md:order-2 space-x-3 md:space-x-0 rtl:space-x-reverse grow justify-end basis-0">
          <Link href={MLFLOW_DOCS_URL} className="hidden md:block">
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
        <div
          className={cn(
            "items-center justify-between w-full md:flex md:w-auto md:order-1 mt-4 md:mt-0",
            isOpen ? "flex" : "hidden md:flex",
          )}
        >
          <ul className="flex flex-col font-medium md:flex-row gap-6 md:gap-10 w-full md:w-auto">
            <li
              className="w-full md:w-auto md:hidden"
              onClick={handleProductItemClick}
            >
              <span
                className={
                  "flex items-center gap-2 py-2 text-white text-[15px] w-full md:w-auto cursor-pointer"
                }
              >
                Components
                <DownIcon className="w-6 h-6" />
              </span>
              <div
                className={cn(
                  "transition-all duration-300 ease-in",
                  isProductSubmenuOpen
                    ? "h-auto min-h-50"
                    : "h-0 overflow-hidden",
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
            <li className="w-full md:w-auto">
              <HeaderMenuItem href={MLFLOW_DOCS_URL} label="Docs" />
            </li>
            <li className="w-full md:w-auto md:hidden">
              <Link href={MLFLOW_DOCS_URL}>
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
    </nav>
  );
};
