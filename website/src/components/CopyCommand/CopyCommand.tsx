import React from "react";

import { Button } from "../Button/Button";

interface Props {
  code: string;
}

export const CopyCommand = ({ code }: Props) => {
  const handleCopy = () => {
    navigator.clipboard.writeText(code);
  };

  return (
    <div className="w-full max-w-3xl mx-auto">
      <div className="bg-white/10 backdrop-blur-[24px] rounded-[32px] p-3 border border-white/20 shadow-lg">
        <div className="bg-white rounded-[20px] p-2 pl-5 border border-white/30">
          <div className="flex items-center justify-between gap-20">
            <div className="flex-1">
              <span className="font-mono text-base text-black">{code}</span>
            </div>
            <Button variant="secondary" size="medium" onClick={handleCopy}>
              Copy to begin
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
