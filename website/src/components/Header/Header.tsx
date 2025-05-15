import { useState, useLayoutEffect } from "react";
import Link from "@docusaurus/Link";
import Logo from "@site/static/img/mlflow-logo-white.svg";
import DownIcon from "@site/static/img/chevron-down-small.svg";

import { cn } from "../../utils";

import { Button } from "../Button/Button";
import { HeaderMenuItem } from "../HeaderMenuItem/HeaderMenuItem";
import { HeaderProductsSubmenu } from "../HeaderProductsSubmenu/HeaderProductsSubmenu";

import "./Header.module.css";

const MD_BREAKPOINT = 640;

export const Header = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isProductItemHovered, setIsProductItemHovered] = useState(false);
  const [isProductSubmenuOpen, setIsProductSubmenuOpen] = useState(false);

  const handleProductItemHover = () => {
    setIsProductItemHovered(true);
  };

  const handleProductSubmenuLeave = () => {
    setIsProductItemHovered(false);
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
    <nav className="fixed w-full z-20 top-0 start-0 bg-[#F7F8F8]/1 border-b border-[#F7F8F8]/8 backdrop-blur-[20px]">
      <div className="flex flex-wrap items-center justify-between mx-auto px-6 md:px-20 py-2 max-w-container">
        <Link
          href="/"
          className="flex items-center space-x-3 rtl:space-x-reverse"
        >
          <Logo className="h-[36px]" />
        </Link>
        <div className="flex flex-row items-center gap-6 md:order-2 space-x-3 md:space-x-0 rtl:space-x-reverse">
          <HeaderMenuItem
            href="https://login.databricks.com/"
            label="Login"
            className="hidden md:block"
          />
          <Link
            href="https://login.databricks.com/?intent=SIGN_UP"
            className="hidden md:block"
          >
            <Button variant="primary" size="small">
              Sign up
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
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
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
                className={cn(
                  "flex items-center gap-2 py-2 text-white text-[15px] w-full md:w-auto cursor-pointer",
                )}
              >
                Products
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
            >
              <HeaderMenuItem label="Product" hasDropdown />
            </li>
            <li className="w-full md:w-auto">
              <HeaderMenuItem href="/releases" label="Releases" />
            </li>
            <li className="w-full md:w-auto">
              <HeaderMenuItem href="/blog" label="Blog" />
            </li>
            <li className="w-full md:w-auto">
              <HeaderMenuItem
                href="https://mlflow.org/docs/latest/"
                label="Docs"
              />
            </li>
            <li className="w-full md:w-auto md:hidden">
              <HeaderMenuItem
                href="https://login.databricks.com/?destination_url=/ml/experiments&dbx_source=MLFLOW_WEBSITE&source=MLFLOW_WEBSITE"
                label="Login"
              />
            </li>
            <li className="w-full md:w-auto md:hidden">
              <Link href="https://signup.databricks.com/?destination_url=/ml/experiments-signup?source=TRY_MLFLOW&dbx_source=TRY_MLFLOW&signup_experience_step=EXPRESS">
                <Button variant="primary" size="small" width="full">
                  Sign up
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
