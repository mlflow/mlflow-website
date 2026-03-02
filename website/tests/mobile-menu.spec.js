"use strict";
var __awaiter =
  (this && this.__awaiter) ||
  function (thisArg, _arguments, P, generator) {
    function adopt(value) {
      return value instanceof P
        ? value
        : new P(function (resolve) {
            resolve(value);
          });
    }
    return new (P || (P = Promise))(function (resolve, reject) {
      function fulfilled(value) {
        try {
          step(generator.next(value));
        } catch (e) {
          reject(e);
        }
      }
      function rejected(value) {
        try {
          step(generator["throw"](value));
        } catch (e) {
          reject(e);
        }
      }
      function step(result) {
        result.done
          ? resolve(result.value)
          : adopt(result.value).then(fulfilled, rejected);
      }
      step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
  };
var __generator =
  (this && this.__generator) ||
  function (thisArg, body) {
    var _ = {
        label: 0,
        sent: function () {
          if (t[0] & 1) throw t[1];
          return t[1];
        },
        trys: [],
        ops: [],
      },
      f,
      y,
      t,
      g = Object.create(
        (typeof Iterator === "function" ? Iterator : Object).prototype,
      );
    return (
      (g.next = verb(0)),
      (g["throw"] = verb(1)),
      (g["return"] = verb(2)),
      typeof Symbol === "function" &&
        (g[Symbol.iterator] = function () {
          return this;
        }),
      g
    );
    function verb(n) {
      return function (v) {
        return step([n, v]);
      };
    }
    function step(op) {
      if (f) throw new TypeError("Generator is already executing.");
      while ((g && ((g = 0), op[0] && (_ = 0)), _))
        try {
          if (
            ((f = 1),
            y &&
              (t =
                op[0] & 2
                  ? y["return"]
                  : op[0]
                    ? y["throw"] || ((t = y["return"]) && t.call(y), 0)
                    : y.next) &&
              !(t = t.call(y, op[1])).done)
          )
            return t;
          if (((y = 0), t)) op = [op[0] & 2, t.value];
          switch (op[0]) {
            case 0:
            case 1:
              t = op;
              break;
            case 4:
              _.label++;
              return { value: op[1], done: false };
            case 5:
              _.label++;
              y = op[1];
              op = [0];
              continue;
            case 7:
              op = _.ops.pop();
              _.trys.pop();
              continue;
            default:
              if (
                !((t = _.trys), (t = t.length > 0 && t[t.length - 1])) &&
                (op[0] === 6 || op[0] === 2)
              ) {
                _ = 0;
                continue;
              }
              if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) {
                _.label = op[1];
                break;
              }
              if (op[0] === 6 && _.label < t[1]) {
                _.label = t[1];
                t = op;
                break;
              }
              if (t && _.label < t[2]) {
                _.label = t[2];
                _.ops.push(op);
                break;
              }
              if (t[2]) _.ops.pop();
              _.trys.pop();
              continue;
          }
          op = body.call(thisArg, _);
        } catch (e) {
          op = [6, e];
          y = 0;
        } finally {
          f = t = 0;
        }
      if (op[0] & 5) throw op[1];
      return { value: op[0] ? op[1] : void 0, done: true };
    }
  };
Object.defineProperty(exports, "__esModule", { value: true });
var test_1 = require("@playwright/test");
test_1.test.describe("Mobile menu accordion behavior", function () {
  test_1.test.beforeEach(function (_a) {
    return __awaiter(void 0, [_a], void 0, function (_b) {
      var page = _b.page;
      return __generator(this, function (_c) {
        switch (_c.label) {
          case 0:
            // Set mobile viewport
            return [
              4 /*yield*/,
              page.setViewportSize({ width: 390, height: 844 }),
            ];
          case 1:
            // Set mobile viewport
            _c.sent();
            return [4 /*yield*/, page.goto("/")];
          case 2:
            _c.sent();
            return [4 /*yield*/, page.waitForLoadState("networkidle")];
          case 3:
            _c.sent();
            return [2 /*return*/];
        }
      });
    });
  });
  (0, test_1.test)("submenus are collapsed by default", function (_a) {
    return __awaiter(void 0, [_a], void 0, function (_b) {
      var menuButton, componentsSubmenu, docsSubmenu;
      var page = _b.page;
      return __generator(this, function (_c) {
        switch (_c.label) {
          case 0:
            menuButton = page.locator(
              'button[data-collapse-toggle="navbar-sticky"]',
            );
            return [4 /*yield*/, menuButton.click()];
          case 1:
            _c.sent();
            componentsSubmenu = page.locator("#mobile-components-submenu");
            docsSubmenu = page.locator("#mobile-docs-submenu");
            return [
              4 /*yield*/,
              (0, test_1.expect)(componentsSubmenu).toHaveClass(/max-h-0/),
            ];
          case 2:
            _c.sent();
            return [
              4 /*yield*/,
              (0, test_1.expect)(docsSubmenu).toHaveClass(/max-h-0/),
            ];
          case 3:
            _c.sent();
            return [2 /*return*/];
        }
      });
    });
  });
  (0, test_1.test)("accordion expands one section at a time", function (_a) {
    return __awaiter(void 0, [_a], void 0, function (_b) {
      var menuButton,
        componentsButton,
        docsButton,
        componentsSubmenu,
        docsSubmenu;
      var page = _b.page;
      return __generator(this, function (_c) {
        switch (_c.label) {
          case 0:
            menuButton = page.locator(
              'button[data-collapse-toggle="navbar-sticky"]',
            );
            return [4 /*yield*/, menuButton.click()];
          case 1:
            _c.sent();
            componentsButton = page.locator(
              'button[aria-controls="mobile-components-submenu"]',
            );
            docsButton = page.locator(
              'button[aria-controls="mobile-docs-submenu"]',
            );
            componentsSubmenu = page.locator("#mobile-components-submenu");
            docsSubmenu = page.locator("#mobile-docs-submenu");
            // Click Components - should expand
            return [4 /*yield*/, componentsButton.click()];
          case 2:
            // Click Components - should expand
            _c.sent();
            return [
              4 /*yield*/,
              (0, test_1.expect)(componentsButton).toHaveAttribute(
                "aria-expanded",
                "true",
              ),
            ];
          case 3:
            _c.sent();
            return [
              4 /*yield*/,
              (0, test_1.expect)(componentsSubmenu).toHaveClass(
                /max-h-\[600px\]/,
              ),
            ];
          case 4:
            _c.sent();
            return [
              4 /*yield*/,
              (0, test_1.expect)(docsSubmenu).toHaveClass(/max-h-0/),
            ];
          case 5:
            _c.sent();
            // Click Docs - Components should collapse, Docs should expand
            return [4 /*yield*/, docsButton.click()];
          case 6:
            // Click Docs - Components should collapse, Docs should expand
            _c.sent();
            return [
              4 /*yield*/,
              (0, test_1.expect)(docsButton).toHaveAttribute(
                "aria-expanded",
                "true",
              ),
            ];
          case 7:
            _c.sent();
            return [
              4 /*yield*/,
              (0, test_1.expect)(componentsButton).toHaveAttribute(
                "aria-expanded",
                "false",
              ),
            ];
          case 8:
            _c.sent();
            return [
              4 /*yield*/,
              (0, test_1.expect)(docsSubmenu).toHaveClass(/max-h-\[300px\]/),
            ];
          case 9:
            _c.sent();
            return [
              4 /*yield*/,
              (0, test_1.expect)(componentsSubmenu).toHaveClass(/max-h-0/),
            ];
          case 10:
            _c.sent();
            return [2 /*return*/];
        }
      });
    });
  });
  (0, test_1.test)("submenus reset when menu closes", function (_a) {
    return __awaiter(void 0, [_a], void 0, function (_b) {
      var menuButton, componentsButton;
      var page = _b.page;
      return __generator(this, function (_c) {
        switch (_c.label) {
          case 0:
            menuButton = page.locator(
              'button[data-collapse-toggle="navbar-sticky"]',
            );
            // Open menu and expand a submenu
            return [4 /*yield*/, menuButton.click()];
          case 1:
            // Open menu and expand a submenu
            _c.sent();
            componentsButton = page.locator(
              'button[aria-controls="mobile-components-submenu"]',
            );
            return [4 /*yield*/, componentsButton.click()];
          case 2:
            _c.sent();
            return [
              4 /*yield*/,
              (0, test_1.expect)(componentsButton).toHaveAttribute(
                "aria-expanded",
                "true",
              ),
            ];
          case 3:
            _c.sent();
            // Close menu
            return [4 /*yield*/, menuButton.click()];
          case 4:
            // Close menu
            _c.sent();
            // Reopen menu - submenu should be collapsed
            return [4 /*yield*/, menuButton.click()];
          case 5:
            // Reopen menu - submenu should be collapsed
            _c.sent();
            return [
              4 /*yield*/,
              (0, test_1.expect)(componentsButton).toHaveAttribute(
                "aria-expanded",
                "false",
              ),
            ];
          case 6:
            _c.sent();
            return [
              4 /*yield*/,
              (0, test_1.expect)(
                page.locator("#mobile-components-submenu"),
              ).toHaveClass(/max-h-0/),
            ];
          case 7:
            _c.sent();
            return [2 /*return*/];
        }
      });
    });
  });
});
