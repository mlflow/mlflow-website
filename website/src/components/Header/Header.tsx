import { useState, useLayoutEffect } from "react";
import Logo from "@site/static/img/mlflow-logo-white.svg";

import { cn } from "../../utils";

import { Button } from "../Button/Button";
import { HeaderMenuItem } from "../HeaderMenuItem/HeaderMenuItem";

import "./Header.module.css";

const MD_BREAKPOINT = 640;

export const Header = () => {
  const [isOpen, setIsOpen] = useState(false);

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
      <div className="flex flex-wrap items-center justify-between mx-auto px-6 md:px-20 py-2">
        <a href="/" className="flex items-center space-x-3 rtl:space-x-reverse">
          <Logo className="h-[36px]" />
        </a>
        <div className="flex flex-row items-center gap-6 md:order-2 space-x-3 md:space-x-0 rtl:space-x-reverse">
          <HeaderMenuItem
            href="https://login.databricks.com/"
            label="Login"
            className="hidden md:block"
          />
          <a
            href="https://login.databricks.com/?intent=SIGN_UP"
            className="hidden md:block"
          >
            <Button variant="primary" size="small">
              Sign up
            </Button>
          </a>
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
            <li className="w-full md:w-auto">
              <HeaderMenuItem href="/product" label="Product" />
            </li>
            <li className="w-full md:w-auto">
              <HeaderMenuItem href="/compare" label="Compare" />
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
                href="https://login.databricks.com/"
                label="Login"
              />
            </li>
            <li className="w-full md:w-auto md:hidden">
              <a href="https://login.databricks.com/?intent=SIGN_UP">
                <Button variant="primary" size="small" width="full">
                  Sign up
                </Button>
              </a>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};
