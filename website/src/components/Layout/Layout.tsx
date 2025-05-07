import { Header } from "../Header/Header";
import { Footer } from "../Footer/Footer";

interface Props {
  children: React.ReactNode;
  variant?: "red" | "blue" | "colorful";
}

export const Layout = ({ children, variant = "red" }: Props) => {
  return (
    <div className="flex flex-col min-h-screen w-full bg-[#0E1416]">
      <Header />
      <main className="flex flex-col">{children}</main>
      <Footer variant={variant} />
    </div>
  );
};
